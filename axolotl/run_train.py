import datetime
import os
import os.path
import gc
from itertools import chain
import random
import time

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
from .model.transformer import DiscreteDiT
from .model.ema import ExponentialMovingAverage

from transformers import PreTrainedTokenizerFast
import esm
from omegaconf import OmegaConf
import wandb
from typing import List
import shutil

torch.backends.cudnn.benchmark = True # TODO: should probably turn off for nested jagged tensors
# torch.autograd.set_detect_anomaly(True)


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    # Set environment variable by reading from secret file
    # This is a good practice to avoid exposing your API key
    # You can also set this in your bashrc or zshrc file
    with open("secret.txt", "r") as f:
        os.environ['WANDB_API_KEY'] = f.read().strip()

    # Set the device for the current process
    torch.cuda.set_device(rank)
    device = torch.device('cuda', rank)
    torch.set_float32_matmul_precision('high')
    # initialize the process group
    dist.init_process_group(
        "nccl", 
        rank=rank, 
        world_size=world_size, 
        timeout=datetime.timedelta(minutes=30),
        device_id=device,
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
    work_dir = config.work_dir

    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")

    if config.train_lora:
        #copy the meta-information to the new work_dir
        print("Copying meta-information for LoRA training")
        shutil.copytree(os.path.join(config.load_dir, "checkpoints-meta"), os.path.join(work_dir, "checkpoints-meta"), dirs_exist_ok=True)

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
                config=OmegaConf.to_container(config),
                id=config.wandb.id or None,
                name=config.wandb.name or None,
                resume='must' if config.wandb.id is not None else None,
            )
            logger.info(f"wandb initiated with run id: {run.id} and run name: {run.name}")
            config.wandb.id = run.id
            config.wandb.name = run.name
            global_table = wandb.Table(columns=["step", "id", "label", "cfg_w", "sampling_steps", "sequence"]) # workaround. TODO: See if the issue gets fixed https://github.com/wandb/wandb/issues/2981
        # save the config to the work directory
        

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
    model = DiscreteDiT(config).to(device)
    if config.train_lora:
        print("Training with LoRA")
        model = utils.setup_lora(model)
    model.compile(mode='default')
    model = DDP(model, device_ids=[rank], static_graph=True) #, find_unused_parameters=True)

    num_parameters = sum(p.numel() for p in model.parameters())
    mprint(f"Number of parameters in the model: {num_parameters}")
    mprint(f"Number of parameters in the model (trainable): {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    ema = ExponentialMovingAverage(
        model.parameters(), decay=config.training.ema)
    mprint(model)
    mprint(f"EMA: {ema}")

    # build noise
    noise = noise_lib.get_noise(config.noise.type, config.noise.sigma_min, config.noise.sigma_max, config.noise.eps).to(device)
    noise = DDP(noise, device_ids=[rank], static_graph=True)
    sampling_eps = 1e-5


    # build optimization state
    optimizer = losses.get_optimizer(config, chain(model.parameters(), noise.parameters()))
    mprint(f"Optimizer: {optimizer}")
    scaler = torch.amp.GradScaler('cuda')
    mprint(f"Scaler: {scaler}")
    state = dict(optimizer=optimizer, scaler=scaler, model=model, noise=noise, ema=ema, step=0) 


    # load in state
    train_lora = config.train_lora or None
    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device, train_lora)
    initial_step = int(state['step'])

    # load in tokenizer
    tokenizer: PreTrainedTokenizerFast = PreTrainedTokenizerFast.from_pretrained(config.data.tokenizer_path)

    # Build data iterators
    train_ds, eval_ds = data.get_dataloaders(config.training.batch_size,
                                             config.eval.batch_size,
                                             config.ngpus,
                                             config.training.accum,
                                             config.data.train_path,
                                             config.data.valid_path,
                                             config.model.length,
                                             config.training.drop_last,
                                             config.training.num_workers,
                                             distributed=True,
                                             seed=42,
                                             epoch=initial_step, # use the initial step as epoch, to make the dataloader shuffle on restart
                                             shuffle_each_epoch=True,
                                             use_unclustered_dataset=config.data.use_unclustered_dataset or False,
    )

    # mprint(f"Length of datasets: {len(train_ds)}, {len(eval_ds)}")

    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(config)
    train_step_fn = losses.get_step_fn(noise, graph, True, optimize_fn, config.training.accum, config.prediction_type, config.training.t_sampling)
    eval_step_fn = losses.get_step_fn(noise, graph, False, optimize_fn, config.training.accum, config.prediction_type, config.training.t_sampling)


    if config.training.snapshot_sampling:
        batch_size = config.eval.batch_size // config.ngpus
        sampling_shape = (batch_size, config.sampling.length)
        sampling_fn = sampling.get_sampling_fn(config, graph, noise, sampling_shape, sampling_eps, device)

    num_train_steps = config.training.n_iters
    mprint(f"Starting training loop at step {initial_step}. The step is used as a seed to shuffle the data.")

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

                mprint("step: %d, training_loss: %.5f" % (step, loss.item()))
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

                mprint("step: %d, evaluation_loss: %.5f" % (step, eval_loss.item()))
                mlog({"evaluation_loss": eval_loss.item()}, step=step)

            if step % config.training.snapshot_freq == 0 or step == num_train_steps:
                # Save the checkpoint.
                save_step = step // config.training.snapshot_freq
                if rank == 0 and config.training.snapshot_checkpoint:
                    utils.save_checkpoint(os.path.join(
                        checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

                # Generate and save samples
                if config.training.snapshot_sampling:
                    mprint(f"Generating text at step: {step}")

                    this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                    utils.makedirs(this_sample_dir)

                    # For LoRA training, don't use ema
                    if config.train_lora:
                        # ema_params = [p for p in model.parameters() if p.requires_grad]
                        # ema.store(ema_params)
                        # ema.copy_to(ema_params)
                        sample, sampling_label, sampling_cfg_w, _, _ = sampling_fn(model)
                        # ema.restore(ema_params)
                    else:
                        ema.store(model.parameters())
                        ema.copy_to(model.parameters())
                        sample, sampling_label, sampling_cfg_w, _, _ = sampling_fn(model)
                        ema.restore(model.parameters())

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
                    

                    if rank == 0:
                        gathered_sequences = list(chain(*sequences_list))
                        sampling_label = torch.cat(label_list)
                        sampling_cfg_w = torch.cat(cfg_w_list)
                        if config.wandb.use_wandb:
                            current_table = wandb.Table(columns=global_table.columns, data=global_table.data) # workaround
                        else:
                            current_table = None
                        steps = config.sampling.steps

                        file_name = os.path.join(this_sample_dir, f"samples.txt")
                        
                        sampling.write_samples(file_name, gathered_sequences, sampling_label, sampling_cfg_w, steps, name='sample', use_wandb=config.wandb.use_wandb, current_table=current_table)
                        mprint(f"Samples saved at step: {step}.")

                        if config.wandb.use_wandb:
                            run.log({"samples": current_table}, step=step)
                            global_table = current_table # workaround

                    if config.eval.perplexity and step != 0:
                        perplexity = calculate_perplexity(config, sequences, device, world_size)
                        mprint(f"Generative Perplexity at step: {step}. Perplexity: {perplexity:.3f}.")
                        mlog({"generative_perplexity": perplexity.item()}, step=step)

                    dist.barrier()


def calculate_perplexity(config, sequences: List[str], device, world_size):
    """
    Calculate the perplexity of the generated sequences using ESM model on a distributed setup.
    Args:
        config: The configuration object.
        sequences: The generated sequences.
        device: The device to use for computation.
        world_size: The number of devices.
    Returns:
        perplexity: The calculated perplexity.
    """
    with torch.no_grad():
        # get the ESM model
        eval_model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        eval_model = eval_model.to(device).eval()

        # convert the sequences to ESM tokens
        sequences = [seq.replace("[", "<cls>").replace("]", "<eos>").replace("?", "X") for seq in sequences]
        esm_sample = torch.cat([torch.tensor(alphabet.encode(seq)).to(device).unsqueeze(0) for seq in sequences], dim=0)

        # batch the samples
        # n_batches = esm_sample.shape[0] // config.eval.perplexity_batch_size
        n_samples = esm_sample.shape[0]
        batch_size = min(config.eval.perplexity_batch_size, n_samples)
        n_batches = max(1, n_samples // batch_size)  # Ensure at least 1 batch
        
        total_nll = torch.tensor(0.0, device=device)
        total_tokens = torch.tensor(0, device=device)

        for batch_idx in range(n_batches):

            # initialize the logits
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            batch = esm_sample[start_idx:end_idx]

            esm_logits = torch.zeros(batch.shape[0], batch.shape[1], len(alphabet)).to(device)
            
            # MLM, mask each token and get logits. Requires many forward passes
            for pos in range(batch.shape[1]):
                batch_masked = batch.clone()
                batch_masked[:, pos] = alphabet.mask_idx
                esm_logits[:, pos, :] = eval_model(batch_masked)["logits"][:, pos, :]

            # calculate negative log-likelihoods
            esm_logits = esm_logits.transpose(1, 2)
            total_nll += F.cross_entropy(esm_logits, batch, reduction="sum")
            total_tokens += batch.shape[0] * batch.shape[1]

        # reduce across all processes
        dist.all_reduce(total_nll)
        dist.all_reduce(total_tokens)

        avg_nll = total_nll / total_tokens
        perplexity = torch.exp(avg_nll)

        del eval_model, esm_logits
        gc.collect()

        return perplexity