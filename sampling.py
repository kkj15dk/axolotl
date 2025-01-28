import abc
import torch
import torch.nn.functional as F
from catsample import sample_categorical

from model import utils as mutils
from typing import Optional

_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

    
def get_predictor(name):
    return _PREDICTORS[name]


def classifier_free_guidance(score, cfg_w):
    """Guidance for classifier-free sampling."""
    
    n = score.shape[0] // 2

    cond_score = score[:n]
    uncond_score = score[n:]

    score = uncond_score + cfg_w * (cond_score - uncond_score)

    return score


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph, noise):
        super().__init__()
        self.graph = graph
        self.noise = noise

    @abc.abstractmethod
    def update_fn(self, score_fn, x, t, step_size, label, cfg_w=None):
        """One update of the predictor.

        Args:
            score_fn: score function
            x: A PyTorch tensor representing the current state
            t: A Pytorch tensor representing the current time step.

        Returns:
            x: A PyTorch tensor of the next state.
        """
        pass


@register_predictor(name="euler")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, label, cfg_w=None):
        sigma, dsigma = self.noise(t)
        score = score_fn(x, sigma, label)

        if cfg_w is not None:
            score = classifier_free_guidance(score, cfg_w)

        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, label, cfg_w=None):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, label, cfg_w=None):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma, label)

        if cfg_w is not None:
            score = classifier_free_guidance(score, cfg_w)

        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return sample_categorical(probs)

    
class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t, label, cfg_w=None):
        sigma = self.noise(t)[0]

        score = score_fn(x, sigma, label)

        if cfg_w is not None:
            score = classifier_free_guidance(score, cfg_w)

        stag_score = self.graph.staggered_score(score, sigma)
        probs = stag_score * self.graph.transp_transition(x, sigma)
        # truncate probabilities
        if self.graph.absorb:
            probs = probs[..., :-1]
        
        #return probs.argmax(dim=-1)
        return sample_categorical(probs)
                       

def get_sampling_fn(config, graph, noise, batch_dims, eps, device):
    
    sampling_fn = get_pc_sampler(graph=graph,
                                 noise=noise,
                                 batch_dims=batch_dims,
                                 predictor=config.sampling.predictor,
                                 steps=config.sampling.steps,
                                 denoise=config.sampling.noise_removal,
                                 eps=eps,
                                 device=device,
                                 cfg=config.sampling.cfg,
                                 label=config.sampling.label,
    )
    
    return sampling_fn
    

def get_pc_sampler(graph, noise, batch_dims, predictor, steps, denoise=True, eps=1e-5, device=torch.device('cpu'), proj_fun=lambda x: x, cfg: int=1, label: str=None):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        if label == 'prokaryotic': # prokaryotic = 0
            input_label = torch.zeros(batch_dims[0], device=device, dtype=torch.long)
        elif label == 'eukaryotic': # eukaryotic = 1
            input_label = torch.ones(batch_dims[0], device=device, dtype=torch.long)
        elif label == 'random':
            input_label = torch.randint(0, 2, (batch_dims[0],), device=device, dtype=torch.long)
        else:
            raise ValueError(f"Invalid label: {label}")
        

        if cfg == 0: # unconditional sampling
            input_label = model.num_labels * torch.ones(batch_dims[0], device=device, dtype=torch.long)
            use_cfg = False
            cfg_w = None
        elif cfg == 1: # conditional sampling
            use_cfg = False # Now need for interpolation at cfg_w = 1
            cfg_w = None
        else: # We are interpolating or extrapolating
            use_cfg = True
            cfg_w = cfg


        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True, use_cfg=use_cfg)
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in range(steps):
            t = timesteps[i] * torch.ones(x.shape[0], 1, device=device)
            x = projector(x)
            x = predictor.update_fn(sampling_score_fn, x, t, dt, input_label, cfg_w)
            

        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], 1, device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t, input_label, cfg_w)
            
        return x, input_label, cfg
    
    return pc_sampler

