import torch

import os
import logging
from omegaconf import OmegaConf, open_dict
import argparse

from peft import get_peft_model, LoraConfig, TaskType

def float_list_or_testing(value):
    if value == 'testing':
        return value
    try:
        # Check if the value is a single float
        if ',' not in value:
            return [float(value)]
        # Otherwise, split the value and convert each part to float
        return [float(v) for v in value.split(',')]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid cfg_w: {value}. Must be a float, a list of floats, or 'testing'.")

def load_hydra_config_from_run(load_dir):
    config_path = os.path.join(load_dir, ".hydra/config.yaml")
    config = OmegaConf.load(config_path)
    return config


def makedirs(dirname):
    os.makedirs(dirname, exist_ok=True)


def get_logger(logpath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    if (logger.hasHandlers()):
        logger.handlers.clear()

    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        info_file_handler.setFormatter(formatter)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


def restore_checkpoint(ckpt_dir, state, device, train_lora=None):
    if not os.path.exists(ckpt_dir):
        makedirs(os.path.dirname(ckpt_dir))
        logging.warning(f"No checkpoint found at {ckpt_dir}. Returned the same state as input")
        return state
    else:
        loaded_state = torch.load(ckpt_dir, map_location=device, weights_only=False) # TODO safe serialization when saving, to do weights_only=True
        
        # Try to load optimizer state, but skip if parameter groups don't match
        try:
            state['optimizer'].load_state_dict(loaded_state['optimizer'])
        except ValueError as e:
            if "parameter group" in str(e):
                logging.warning(f"Skipping optimizer state loading due to parameter group mismatch: {e}")
                logging.warning("This is expected when switching between LoRA and full fine-tuning")
            else:
                raise e
        
        state['model'].module.load_state_dict(loaded_state['model'], strict=False)
        
        # Reinitialize EMA when using LoRA to avoid parameter mismatch
        if train_lora:
            logging.warning("Reinitializing EMA because LoRA training is enabled")
            logging.warning("This ensures compatibility when switching between LoRA and full fine-tuning")
            # Reinitialize EMA with current model parameters
            from .model.ema import ExponentialMovingAverage
            decay = loaded_state['ema']['decay']
            state['ema'] = ExponentialMovingAverage(state['model'].parameters(), decay=decay)
        else:
            state['ema'].load_state_dict(loaded_state['ema'])
        
        state['step'] = loaded_state['step']
        return state


def save_checkpoint(ckpt_dir, state):
    saved_state = {
        'optimizer': state['optimizer'].state_dict(),
        'model': state['model'].module.state_dict(),
        'ema': state['ema'].state_dict(),
        'step': state['step']
    }
    torch.save(saved_state, ckpt_dir)


def setup_lora(model):
    peft_config = LoraConfig(
        target_modules=["attn_qkv", "attn_out", "linear"], # TODO? , 'mlp', timestep embedding, positional embedding, class embedding
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        bias='none',
    )

    peft_model = get_peft_model(model, peft_config)
    peft_model.print_trainable_parameters()

    return peft_model