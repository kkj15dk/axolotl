# %%
import datetime
import os
import os.path
import gc
from itertools import chain
import random

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

import data
import losses
import sampling
import graph_lib
import noise_lib
import utils
from model import SEDD_flex
from model.ema import ExponentialMovingAverage
from transformers import GPT2TokenizerFast, GPT2LMHeadModel


torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30)
    )


def cleanup():
    dist.destroy_process_group()


def run_multiprocess(rank, world_size, cfg, port):
    try:
        setup(rank, world_size, port)
        _run(rank, world_size, cfg)
    finally:
        cleanup()

def get_random_batch(cfg):
    # get random data
    batch_size = cfg.training.batch_size
    
    # Current limitation that the total sequnce length must be divisible by 128
    sentence_lengths = [random.randint(1, 1024) for _ in range(batch_size - 1)]
    print("sentence lengths:", sentence_lengths)
    total = sum(sentence_lengths)
    sentence_lengths.append(128 - total % 128)
    total = sum(sentence_lengths)
    print("actual total:", total)
    
    start = [random.randint(0, cfg.tokens - l) for l in sentence_lengths]
    ragged_tensors = [torch.arange(start[i], start[i] + l, device="cuda") for i, l in enumerate(sentence_lengths)]
    batch = torch.nested.nested_tensor(
        ragged_tensors, layout=torch.jagged
    )
    return batch

def _run(rank, world_size, cfg):
    torch.cuda.set_device(rank)
    work_dir = cfg.work_dir

    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    if rank == 0:
        utils.makedirs(sample_dir)
        utils.makedirs(checkpoint_dir)
        utils.makedirs(os.path.dirname(checkpoint_meta_dir))

    # logging
    if rank == 0:
        logger = utils.get_logger(os.path.join(work_dir, "logs"))
    def mprint(msg):
        if rank == 0:
            logger.info(msg)

    mprint(work_dir)
    mprint(cfg)
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        mprint("Found {} CUDA devices.".format(torch.cuda.device_count()))
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mprint(
                "{} \t Memory: {:.2f}GB".format(
                    props.name, props.total_memory / (1024 ** 3)
                )
            )
    else:
        mprint("WARNING: Using device {}".format(device))
    mprint(f"Found {os.cpu_count()} total number of CPUs.")

    # build token graph
    graph = graph_lib.get_graph(cfg, device)
    
    # build score model
    score_model = SEDD_flex(cfg).to(device)
    score_model = DDP(score_model, device_ids=[rank], static_graph=True, find_unused_parameters=True)

    num_parameters = sum(p.numel() for p in score_model.parameters())
    mprint(f"Number of parameters in the model: {num_parameters}")

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=cfg.training.ema)
    mprint(score_model)
    mprint(f"EMA: {ema}")

    # build noise
    noise = noise_lib.get_noise(cfg).to(device)
    noise = DDP(noise, device_ids=[rank], static_graph=True)
    sampling_eps = 1e-5


    # build optimization state
    optimizer = losses.get_optimizer(cfg, chain(score_model.parameters(), noise.parameters()))
    mprint(f"Optimizer: {optimizer}")
    scaler = torch.amp.GradScaler('cuda')
    mprint(f"Scaler: {scaler}")
    state = dict(optimizer=optimizer, scaler=scaler, model=score_model, noise=noise, ema=ema, step=0) 


    # load in state
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    initial_step = int(state['step'])

    
    # load in tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(cfg)
    train_step_fn = losses.get_step_fn(noise, graph, True, optimize_fn, cfg.training.accum)
    eval_step_fn = losses.get_step_fn(noise, graph, False, optimize_fn, cfg.training.accum)


    if cfg.training.snapshot_sampling: # TODO: support for different length sampling
        sampling_shape = (cfg.training.batch_size // (cfg.ngpus * cfg.training.accum), cfg.model.length)
        sampling_fn = sampling.get_sampling_fn(cfg, graph, noise, sampling_shape, sampling_eps, device)

    num_train_steps = cfg.training.n_iters
    mprint(f"Starting training loop at step {initial_step}.")


    while state['step'] < num_train_steps + 1:
        step = state['step']

        batch = get_random_batch(cfg)
        print("batch shape:", batch.shape)

        loss = train_step_fn(state, batch)

        # flag to see if there was movement ie a full batch got computed
        if step != state['step']:
            if step % cfg.training.log_freq == 0:
                dist.all_reduce(loss)
                loss /= world_size

                mprint("step: %d, training_loss: %.5e" % (step, loss.item()))
            
            if step % cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
                utils.save_checkpoint(checkpoint_meta_dir, state)

            if step % cfg.training.eval_freq == 0:
                    
                eval_batch = get_random_batch(cfg)
                eval_loss = eval_step_fn(state, eval_batch)

                dist.all_reduce(eval_loss)
                eval_loss /= world_size

                mprint("step: %d, evaluation_loss: %.5e" % (step, eval_loss.item()))

            if step > 0 and step % cfg.training.snapshot_freq == 0 or step == num_train_steps:
                # Save the checkpoint.
                save_step = step // cfg.training.snapshot_freq
                if rank == 0:
                    utils.save_checkpoint(os.path.join(
                        checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

                # Generate and save samples
                if cfg.training.snapshot_sampling:
                    mprint(f"Generating text at step: {step}")

                    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                    utils.makedirs(this_sample_dir)

                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())
                    sample = sampling_fn(score_model)
                    ema.restore(score_model.parameters())

                    sentences = tokenizer.batch_decode(sample)
                    
                    file_name = os.path.join(this_sample_dir, f"sample_{rank}.txt")
                    with open(file_name, 'w') as file:
                        for sentence in sentences:
                            file.write(sentence + "\n")
                            file.write("============================================================================================\n")

                    if cfg.eval.perplexity:
                        with torch.no_grad():
                            eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
                            batches = sample.shape[0] // cfg.eval.perplexity_batch_size
                            total_perplexity = 0
                            for i in range(batches):
                                s = sample[i * cfg.eval.perplexity_batch_size:(i + 1) * cfg.eval.perplexity_batch_size]
                                loss, logits = eval_model(s, labels=s)[:2]
                                logits = logits.transpose(-1, -2)
                                perplexity = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none").mean(dim=-1).exp().mean()
                                total_perplexity += perplexity
                            total_perplexity /= batches
                            dist.all_reduce(total_perplexity)
                            total_perplexity /= world_size
                            mprint(f"Generative Perplexity at step: {step}. Perplexity: {total_perplexity:.3f}.")

                            del eval_model, logits, loss

                    dist.barrier()


# %%
"""Training and evaluation"""

import hydra
import os
import numpy as np
import utils
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from omegaconf import OmegaConf, open_dict

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg):
    ngpus = cfg.ngpus
    if "load_dir" in cfg:
        hydra_cfg_path = os.path.join(cfg.load_dir, ".hydra/hydra.yaml")
        hydra_cfg = OmegaConf.load(hydra_cfg_path).hydra

        cfg = utils.load_hydra_config_from_run(cfg.load_dir)
        
        work_dir = cfg.work_dir
        utils.makedirs(work_dir)
    else:
        hydra_cfg = HydraConfig.get()
        work_dir = hydra_cfg.run.dir if hydra_cfg.mode == RunMode.RUN else os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
        utils.makedirs(work_dir)

    with open_dict(cfg):
        cfg.ngpus = ngpus
        cfg.work_dir = work_dir
        cfg.wandb_name = os.path.basename(os.path.normpath(work_dir))

	# Run the training pipeline
    port = int(np.random.randint(10000, 20000))
    logger = utils.get_logger(os.path.join(work_dir, "logs"))

    hydra_cfg = HydraConfig.get()
    if hydra_cfg.mode != RunMode.RUN:
        logger.info(f"Run id: {hydra_cfg.job.id}")

    try:
        mp.set_start_method("forkserver")
        mp.spawn(run_multiprocess, args=(ngpus, cfg, port), nprocs=ngpus, join=True)
    except Exception as e:
        logger.critical(e, exc_info=True)


if __name__ == "__main__":
    main()


