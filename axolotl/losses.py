import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from . import graph_lib
from .model import utils as mutils


def get_loss_fn(noise, graph: graph_lib.Graph, train, prediction_type='log_score', t_sampling='uniform', sampling_eps=1e-3):

    def loss_fn(model, input_ids, label):
        """
        Batch shape: [B, L] int. D given from graph
        """

        bs = input_ids.shape[0]

        if t_sampling == 'uniform':
            t = torch.rand(bs, device=input_ids.device)
        elif t_sampling == 'antithetic':
            t0 = torch.rand((1,), device=input_ids.device).item()
            t = torch.remainder(t0 + torch.arange(start=0, end=1, step=1/bs, device=input_ids.device), 1)
        alpha = noise(t, alpha=True)
        
        perturbed_batch = graph.sample_transition(input_ids, alpha[:, None])

        if prediction_type == 'log_score':
            dbeta = noise(t, dbeta=True)

            log_score_fn = mutils.get_output_fn(model, train=train, exponentiate=False)
            log_score = log_score_fn(perturbed_batch, t, label)
            loss = graph.score_entropy(log_score, t[:, None], perturbed_batch, input_ids) # loss shape: (B, j1)

            if log_score.is_nested:
                loss = (dbeta[:, None] * loss)
            else:
                raise NotImplementedError("Not implemented yet, shouldn't use sum if i use mean for nested tensor loss")
                loss = (dbeta[:, None] * loss).sum(dim=-1)

        elif prediction_type == 'x0':
            alpha_t1 = noise(t=0.0, alpha=True)
            dgamma_times_alpha = noise(t, dgamma_times_alpha=True)

            logits_fn = mutils.get_output_fn(model, train=train, exponentiate=False)
            logits = logits_fn(perturbed_batch, t, label)

            loss = graph.x0_entropy(logits, alpha_t1, dgamma_times_alpha[:, None], perturbed_batch, input_ids) # loss shape: (B, j1)
        
        else:
            raise NotImplementedError(f"Prediction type {prediction_type} not implemented yet!")

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


def get_step_fn(noise, graph, train, optimize_fn, accum, prediction_type, t_sampling):
    loss_fn = get_loss_fn(noise, graph, train, prediction_type, t_sampling)

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