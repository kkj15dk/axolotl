import datetime
import os
import os.path
import gc
from itertools import chain
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from . import (
    data_nested as data, 
    losses, 
    sampling,
    graph_lib,
    noise_lib,
    utils
)
from .model.transformer import SEDD
from .model.ema import ExponentialMovingAverage

from transformers import PreTrainedTokenizerFast
import esm
from omegaconf import OmegaConf
import wandb

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
    wandb.finish()

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


    # logging TODO make sure restarting works for wandb
    if rank == 0:
        logger = utils.get_logger(os.path.join(work_dir, "logs"))
        if config.wandb.use_wandb:
            run = wandb.init(
                entity=config.wandb.entity,
                project=config.wandb.project,
                config=OmegaConf.to_container(config)
            )
            logger.info(f"wandb initiated with run id: {run.id} and run name: {run.name}")
            global_table = wandb.Table(columns=["step", "id", "label", "cfg_w", "sampling_steps", "sequence"]) # workaround. TODO: See if the issue gets fixed https://github.com/wandb/wandb/issues/2981

    def mprint(msg):
        if rank == 0:
            logger.info(msg)
    def mlog(dict, step=None):
        if rank == 0 and config.wandb.use_wandb:
            run.log(dict, step=step)
    
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
    score_model = DDP(score_model, device_ids=[rank], static_graph=True) #, find_unused_parameters=True)

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
    tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(config.data.tokenizer_path)

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


    if config.training.snapshot_sampling:
        batch_size = config.eval.batch_size // config.ngpus
        sampling_shape = (batch_size, config.sampling.length)
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
                    
                    if rank == 0:
                        sequences_list = [[None for i in sequences] for _ in range(world_size)]
                        label_list = [torch.zeros_like(sampling_label) for _ in range(world_size)]
                        cfg_w_list = [torch.zeros_like(sampling_cfg_w) for _ in range(world_size)]
                    else:
                        sequences_list = None
                        label_list = None
                        cfg_w_list = None
                    
                    # gather the samples on rank 0
                    dist.gather_object(sequences, sequences_list, dst=0)
                    dist.gather(sampling_label, label_list, dst=0)
                    dist.gather(sampling_cfg_w, cfg_w_list, dst=0)
                    

                    if rank == 0 and config.wandb.use_wandb:
                        gathered_sequences = list(chain(*sequences_list))
                        sampling_label = torch.cat(label_list)
                        sampling_cfg_w = torch.cat(cfg_w_list)
                        current_table = wandb.Table(columns=global_table.columns, data=global_table.data) # workaround

                        file_name = os.path.join(this_sample_dir, f"samples.txt")
                        with open(file_name, 'w') as file:
                            for i, seq in enumerate(gathered_sequences):
                                if sampling_label[i] == 0:
                                    sequence_label = "prokaryotic"
                                elif sampling_label[i] == 1:
                                    sequence_label = "eukaryotic"
                                else:
                                    raise ValueError(f"Invalid label: {sampling_label[i]}")
                                w = sampling_cfg_w[i].item()
                                
                                file.write(f">{i} label:{sequence_label} cfg_w:{w} sampling_steps:{config.sampling.steps}\n")
                                file.write(seq + "\n")
                            
                                current_table.add_data(step, i, sequence_label, w, config.sampling.steps, seq) # workaround
                        run.log({"samples": current_table}, step=step)
                        global_table = current_table # workaround

                    if config.eval.perplexity:
                        with torch.no_grad():
                            # get the ESM model
                            eval_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
                            eval_model = eval_model.to(device).eval()

                            # convert the sequences to ESM tokens
                            sequences = [seq.replace("[", "<cls>").replace("]", "<eos>").replace("?", "X") for seq in sequences]
                            esm_sample = torch.cat([torch.tensor(alphabet.encode(seq)).to(device).unsqueeze(0) for seq in sequences], dim=0)

                            # batch the samples
                            n_batches = esm_sample.shape[0] // config.eval.perplexity_batch_size
                            total_perplexity = 0
                            for i in range(n_batches):

                                # initialize the logits
                                batch = esm_sample[i * config.eval.perplexity_batch_size:(i + 1) * config.eval.perplexity_batch_size]
                                esm_logits = torch.zeros(batch.shape[0], batch.shape[1], len(alphabet)).to(device)
                                
                                # MLM, mask each token and get logits. Requires many forward passes
                                for i in range(batch.shape[1]):
                                    batch_masked = batch.clone()
                                    batch_masked[:, i] = alphabet.mask_idx
                                    batch_masked = batch_masked.to(device)
                                    esm_logits[:, i, :] = eval_model(batch_masked)["logits"][:, i, :]

                                # calculate perplexity TODO: Check if this is correct
                                esm_logits = esm_logits.transpose(1, 2)
                                perplexity = F.cross_entropy(esm_logits, batch, reduction="none").mean(dim=-1).exp().mean()
                                total_perplexity += perplexity

                            total_perplexity /= n_batches
                            dist.all_reduce(total_perplexity)
                            total_perplexity /= world_size
                            mprint(f"Generative Perplexity at step: {step}. Perplexity: {total_perplexity:.3f}.")
                            mlog({"generative_perplexity": total_perplexity.item()}, step=step)

                            del eval_model, esm_logits

                    dist.barrier()