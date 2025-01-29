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

import data_nested as data
import losses
import sampling
import graph_lib
import noise_lib
import utils
from model import SEDD
from model.ema import ExponentialMovingAverage
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, PreTrainedTokenizerFast
import wandb
from omegaconf import OmegaConf

torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    # Set environment variable by reading from secret file
    # This is a good practice to avoid exposing your API key
    # You can also set this in your bashrc or zshrc file
    with open("secret.txt", "r") as f:
        os.environ['WANDB_API_KEY'] = f.read().strip()

    # initialize the process group
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30)
    )


def cleanup():
    dist.destroy_process_group()


def run_multiprocess(rank, world_size, config, port):
    try:
        setup(rank, world_size, port)
        _run(rank, world_size, config)
    finally:
        cleanup()


def _run(rank, world_size, config):
    torch.cuda.set_device(rank)
    work_dir = config.work_dir

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
        run = wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config)
        )

    def mprint(msg):
        if rank == 0:
            logger.info(msg)
    def mlog(dict, step):
        if rank == 0:
            run.log(dict, step=step)
    
    mprint(f"wandb initiated with run id: {run.id} and run name: {run.name}")
    mprint(work_dir)
    mprint(config)
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

    # CUDA version
    mprint('CUDA version: {}'.format(torch.version.cuda))
    # mprint('CUDA_PATH: {}'.format(os.environ['CUDA_PATH']))
    # mprint('CUDA_HOME: {}'.format(os.environ['CUDA_HOME']))

    # build token graph
    graph = graph_lib.get_graph(config, device)
    
    # build score model
    score_model = SEDD(config).to(device)
    score_model = DDP(score_model, device_ids=[rank], static_graph=True, find_unused_parameters=True)

    num_parameters = sum(p.numel() for p in score_model.parameters())
    mprint(f"Number of parameters in the model: {num_parameters}")

    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=config.training.ema)
    mprint(score_model)
    mprint(f"EMA: {ema}")

    # build noise
    noise = noise_lib.get_noise(config).to(device)
    noise = DDP(noise, device_ids=[rank], static_graph=True)
    sampling_eps = 1e-5


    # build optimization state
    optimizer = losses.get_optimizer(config, chain(score_model.parameters(), noise.parameters()))
    mprint(f"Optimizer: {optimizer}")
    scaler = torch.amp.GradScaler('cuda')
    mprint(f"Scaler: {scaler}")
    state = dict(optimizer=optimizer, scaler=scaler, model=score_model, noise=noise, ema=ema, step=0) 


    # load in state
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    initial_step = int(state['step'])

    # load in tokenizer
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.data.tokenizer_path)

    # Build data iterators
    train_ds, eval_ds = data.get_dataloaders(config)

    # mprint(f"Length of datasets: {len(train_ds)}, {len(eval_ds)}")

    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # # Test
    # batch = next(eval_iter)['input_ids'].to(device)
    # decoded = tokenizer.batch_decode(batch)
    # for i, seq in enumerate(decoded):
    #     length = len(seq)
    #     print(i)
    #     print("length", length)
    #     print(seq)

    # raise ValueError("Test")

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    train_step_fn = losses.get_step_fn(noise, graph, True, optimize_fn, config.training.accum)
    eval_step_fn = losses.get_step_fn(noise, graph, False, optimize_fn, config.training.accum)


    if config.training.snapshot_sampling: # TODO: support for different length sampling
        sampling_shape = (config.training.batch_size // (config.ngpus * config.training.accum * 2), config.sampling.length)
        sampling_fn = sampling.get_sampling_fn(config, graph, noise, sampling_shape, sampling_eps, device)

    num_train_steps = config.training.n_iters
    mprint(f"Starting training loop at step {initial_step}.")


    while state['step'] < num_train_steps + 1:
        step = state['step']

        batch = next(train_iter)
        input_ids = batch['input_ids'].to(device)
        label = batch['label'].to(device)

        loss = train_step_fn(state, input_ids, label)

        # flag to see if there was movement ie a full batch got computed
        if step != state['step']:
            if step % config.training.log_freq == 0:
                dist.all_reduce(loss)
                loss /= world_size

                mprint("step: %d, training_loss: %.5e" % (step, loss.item()))
                mlog({"training_loss": loss.item()}, step=step)
            
            if step % config.training.snapshot_freq_for_preemption == 0 and rank == 0:
                utils.save_checkpoint(checkpoint_meta_dir, state)

            if step % config.training.eval_freq == 0:

                eval_batch = next(eval_iter)
                eval_input_ids = eval_batch['input_ids'].to(device)
                eval_label = eval_batch['label'].to(device)

                eval_loss = eval_step_fn(state, eval_input_ids, eval_label)

                dist.all_reduce(eval_loss)
                eval_loss /= world_size

                mprint("step: %d, evaluation_loss: %.5e" % (step, eval_loss.item()))
                mlog({"evaluation_loss": eval_loss.item()}, step=step)

            if step > 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
                # Save the checkpoint.
                save_step = step // config.training.snapshot_freq
                if rank == 0:
                    utils.save_checkpoint(os.path.join(
                        checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

                # Generate and save samples
                if config.training.snapshot_sampling:
                    mprint(f"Generating text at step: {step}")

                    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                    utils.makedirs(this_sample_dir)

                    ema.store(score_model.parameters())
                    ema.copy_to(score_model.parameters())
                    sample, sampling_label, sampling_cfg_w = sampling_fn(score_model)
                    ema.restore(score_model.parameters())

                    sequences = tokenizer.batch_decode(sample)
                    
                    file_name = os.path.join(this_sample_dir, f"sample_{rank}.txt")
                    with open(file_name, 'w') as file:
                        for i, seq in enumerate(sequences):
                            if sampling_label[i] == 0:
                                sequence_label = "prokaryotic"
                            elif sampling_label[i] == 1:
                                sequence_label = "eukaryotic"
                            else:
                                raise ValueError(f"Invalid label: {sampling_label[i]}")
                            if isinstance(sampling_cfg_w, torch.Tensor):
                                w = sampling_cfg_w[i].item()
                            else:
                                w = sampling_cfg_w
                            file.write(f">{i} | label: {sequence_label} | cfg_w: {w}\n")
                            file.write(seq + "\n")

                    if config.eval.perplexity:
                        with torch.no_grad():
                            eval_model = GPT2LMHeadModel.from_pretrained("gpt2-large").to(device).eval()
                            batches = sample.shape[0] // config.eval.perplexity_batch_size
                            total_perplexity = 0
                            for i in range(batches):
                                s = sample[i * config.eval.perplexity_batch_size:(i + 1) * config.eval.perplexity_batch_size]
                                loss, logits = eval_model(s, labels=s)[:2]
                                logits = logits.transpose(-1, -2)
                                perplexity = F.cross_entropy(logits[..., :-1], s[..., 1:], reduction="none").mean(dim=-1).exp().mean()
                                total_perplexity += perplexity
                            total_perplexity /= batches
                            dist.all_reduce(total_perplexity)
                            total_perplexity /= world_size
                            mprint(f"Generative Perplexity at step: {step}. Perplexity: {total_perplexity:.3f}.")
                            mlog({"generative_perplexity": total_perplexity.item()}, step=step)

                            del eval_model, logits, loss

                    dist.barrier()