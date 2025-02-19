import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from . import graph_lib
from .model import utils as mutils


def get_loss_fn(noise, graph: graph_lib.Graph, train, sampling_eps=1e-3, lv=False):

    def loss_fn(model, input_ids, label, t=None, perturbed_batch=None):
        """
        Batch shape: [B, L] int. D given from graph
        """

        if t is None:
            if lv:
                raise NotImplementedError("Yeah I gotta do this later")
            else:
                t = (1 - sampling_eps) * torch.rand(input_ids.shape[0], device=input_ids.device) + sampling_eps
            
        sigma, dsigma = noise(t)
        
        if perturbed_batch is None:
            perturbed_batch = graph.sample_transition(input_ids, sigma[:, None])

        log_score_fn = mutils.get_score_fn(model, train=train, sampling=False)
        log_score = log_score_fn(perturbed_batch, sigma, label)
        loss = graph.score_entropy(log_score, sigma[:, None], perturbed_batch, input_ids) # loss shape: (B, j1)

        if log_score.is_nested:
            loss = (dsigma[:, None] * loss)
        else:
            raise NotImplementedError("Not implemented yet, shouldn't use sum if i use mean for nested tensor loss")
            loss = (dsigma[:, None] * loss).sum(dim=-1)

        return loss

    return loss_fn


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, 
                    scaler, 
                    params, 
                    step, 
                    lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        scaler.step(optimizer)
        scaler.update()

    return optimize_fn


def get_step_fn(noise, graph, train, optimize_fn, accum):
    loss_fn = get_loss_fn(noise, graph, train)

    accum_iter = 0
    total_loss = 0

    def step_fn(state, input_ids, label):
        nonlocal accum_iter 
        nonlocal total_loss

        model = state['model']
        # def forward_hook(module, input, output):
        #     try:
        #         print(f"Forward hook for {module.__class__.__name__}: input shape {input[0].shape}, output shape {output.shape}")
        #     except:
        #         print(f"{module.__class__.__name__}: can't print shapes")
        # def backward_hook(module, grad_input, grad_output):
        #     try:
        #         print(f"Backward hook for {module.__class__.__name__}: grad_input shape {grad_input[0].shape}, grad_output shape {grad_output[0].shape}")
        #     except:
        #         print(f"{module.__class__.__name__}: can't print shapes")

        # # Register hooks
        # for name, module in model.named_modules():
        #     module.register_forward_hook(forward_hook)
        #     module.register_backward_hook(backward_hook)

        if train:
            optimizer = state['optimizer']
            scaler = state['scaler']
            loss = loss_fn(model, input_ids, label).mean() / accum

            # graph = make_dot(loss_fn(model, batch, cond=cond).mean() / accum, params=dict(model.named_parameters()))
            # graph.render("graph")

            # time1 = timeit.default_timer()
            scaler.scale(loss).backward()
            # time2 = timeit.default_timer()
            # print("time to call backwards", time2 - time1)

            accum_iter += 1
            total_loss += loss.detach()
            if accum_iter == accum:
                accum_iter = 0

                state['step'] += 1
                optimize_fn(optimizer, scaler, model.parameters(), step=state['step'])
                state['ema'].update(model.parameters())
                optimizer.zero_grad()
                
                loss = total_loss
                total_loss = 0
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, input_ids, label).mean()
                ema.restore(model.parameters())

        return loss

    return step_fn