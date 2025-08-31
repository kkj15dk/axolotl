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
        
        # Validate that all loaded parameters match the current model structure
        model_state_dict = state['model'].module.state_dict()
        loaded_model_state = loaded_state['model']
        
        # Handle LoRA model structure differences
        if train_lora:
            # When using LoRA, we need to map checkpoint keys to match the PEFT model structure
            mapped_loaded_state = {}
            for key, value in loaded_model_state.items():
                # Map checkpoint keys to PEFT model keys
                if key.startswith('blocks.') and '.attn_qkv.weight' in key:
                    new_key = f"base_model.model.{key.replace('.attn_qkv.weight', '.attn_qkv.base_layer.weight')}"
                elif key.startswith('blocks.') and '.attn_out.weight' in key:
                    new_key = f"base_model.model.{key.replace('.attn_out.weight', '.attn_out.base_layer.weight')}"
                elif key.startswith('output_layer.linear.weight'):
                    new_key = f"base_model.model.{key.replace('output_layer.linear.weight', 'output_layer.linear.base_layer.weight')}"
                elif key.startswith('output_layer.linear.bias'):
                    new_key = f"base_model.model.{key.replace('output_layer.linear.bias', 'output_layer.linear.base_layer.bias')}"
                else:
                    new_key = f"base_model.model.{key}"
                
                if new_key in model_state_dict:
                    if model_state_dict[new_key].shape == value.shape:
                        mapped_loaded_state[new_key] = value
                    else:
                        raise ValueError(f"Shape mismatch for parameter '{key}' -> '{new_key}': "
                                       f"current model has {model_state_dict[new_key].shape}, "
                                       f"checkpoint has {value.shape}")
                else:
                    raise ValueError(f"Parameter '{key}' -> '{new_key}' exists in checkpoint but not in current model")
            
            loaded_model_state = mapped_loaded_state
        
        # Check for parameter mismatches
        for key, value in loaded_model_state.items():
            if key not in model_state_dict:
                raise ValueError(f"Parameter '{key}' exists in checkpoint but not in current model")
            if model_state_dict[key].shape != value.shape:
                raise ValueError(f"Shape mismatch for parameter '{key}': "
                               f"current model has {model_state_dict[key].shape}, "
                               f"checkpoint has {value.shape}")
        
        for key in model_state_dict.keys():
            if key not in loaded_model_state:
                # Skip LoRA-specific parameters that won't be in the base checkpoint
                if train_lora and ('lora_A' in key or 'lora_B' in key):
                    continue
                raise ValueError(f"Parameter '{key}' exists in current model but not in checkpoint")
        
        # If we get here, all parameters match - safe to load
        state['model'].module.load_state_dict(loaded_model_state, strict=False)
        
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