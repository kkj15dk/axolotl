import abc
import torch
import torch.nn.functional as F
from .catsample import sample_categorical

from .model import utils as mutils
from typing import Optional, Union
from tqdm import tqdm
from typing import List

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


def write_samples(output: str, sequences: List[str], sampling_label: List[str], sampling_cfg_w: torch.FloatTensor, steps: int, name: str = "sample"):
    """Write samples to a file."""
    
    assert sampling_cfg_w.dim() == 1, f"cfg_w must be a 1D tensor, got {sampling_cfg_w.dim()}"
    assert len(sequences) == len(sampling_label) == len(sampling_cfg_w), f"Length mismatch: {len(sequences)}, {len(sampling_label)}, {len(sampling_cfg_w)}"

    print(f"Writing samples to {output}")
    with open(output, "a") as file:
        for i, seq in enumerate(sequences):
            if sampling_label[i] == 0:
                sequence_label = "prokaryotic"
            elif sampling_label[i] == 1:
                sequence_label = "eukaryotic"
            else:
                raise ValueError(f"Invalid label: {sampling_label[i]}")
            w = sampling_cfg_w[i].item()
            
            file.write(f">{name}_{i} label:{sequence_label} cfg_w:{w} steps:{steps}\n")
            file.write(seq + "\n")


def classifier_free_guidance(score, cfg_w: torch.Tensor):
    """Guidance for classifier-free sampling."""
    
    assert isinstance(cfg_w, torch.Tensor), f'cfg_w must be a tensor, got {type(cfg_w)}'

    n = score.shape[0] // 2
    assert cfg_w.dim() == 1, f'cfg_w must be a 1D tensor, got {cfg_w.dim()}'
    assert cfg_w.shape[0] == n, f'cfg_w must have length {n}, got {cfg_w.shape[0]}'
    cfg_w = cfg_w.unsqueeze(-1).unsqueeze(-1)

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
    def update_fn(self, score_fn, x, t, step_size, label, cfg_w, use_cfg=False):
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
    def update_fn(self, score_fn, x, t, step_size, label, cfg_w, use_cfg=False):
        sigma, dsigma = self.noise(t)

        score = score_fn(x, sigma, label)

        if use_cfg:
            score = classifier_free_guidance(score, cfg_w)

        dsigma = dsigma.unsqueeze(-1) # TODO: make it so this is not necessary
        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        x = self.graph.sample_rate(x, rev_rate)
        return x

@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, label, cfg_w):
        return x


@register_predictor(name="analytic")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, label, cfg_w, use_cfg=False):
        curr_sigma = self.noise(t)[0]
        next_sigma = self.noise(t - step_size)[0]
        dsigma = curr_sigma - next_sigma

        score = score_fn(x, curr_sigma, label)

        if use_cfg:
            score = classifier_free_guidance(score, cfg_w)

        dsigma = dsigma.unsqueeze(-1) # TODO: make it so this is not necessary
        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return sample_categorical(probs)

    
class Denoiser:
    def __init__(self, graph, noise):
        self.graph = graph
        self.noise = noise

    def update_fn(self, score_fn, x, t, label, cfg_w, use_cfg=False):
        sigma = self.noise(t)[0]

        score = score_fn(x, sigma, label)

        if use_cfg:
            score = classifier_free_guidance(score, cfg_w)

        sigma = sigma.unsqueeze(-1) # TODO: make it so this is not necessary
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
                                 num_labels=config.num_labels,
    )
    
    return sampling_fn
    

def get_pc_sampler(graph, 
                   noise, 
                   batch_dims, 
                   predictor, 
                   steps, 
                   denoise=True, 
                   eps=1e-5, 
                   device=torch.device('cpu'), 
                   proj_fun=lambda x: x, 
                   cfg: Union[float, List[float], str]=1.0, 
                   label: str=None, 
                   num_labels: int=2,
                   use_tqdm: bool=False
):
    predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise)

    @torch.no_grad()
    def pc_sampler(model):
        batch_size = batch_dims[0]
        if label == 'prokaryotic': # prokaryotic = 0
            input_label = torch.zeros(batch_size, device=device, dtype=torch.long)
        elif label == 'eukaryotic': # eukaryotic = 1
            input_label = torch.ones(batch_size, device=device, dtype=torch.long)
        elif label == 'random' or label is None:
            input_label = torch.randint(0, num_labels, (batch_size,), device=device, dtype=torch.long)
        else:
            raise ValueError(f"Invalid label: {label}")
        
        cfg_w = cfg # cfg weight

        if cfg_w == 0: # unconditional sampling
            input_label = num_labels * torch.ones(batch_size, device=device, dtype=torch.long)
            use_cfg = False
            cfg_w = torch.zeros(batch_size, device=device)
        elif cfg_w == 1: # conditional sampling
            use_cfg = False # Now need for interpolation at cfg_w = 1
            cfg_w = torch.ones(batch_size, device=device)
        else: # We are interpolating or extrapolating
            use_cfg = True
            if cfg_w == 'testing':
                if batch_size == 1:
                    cfg_w = torch.tensor([1], device=device)
                elif batch_size == 2:
                    cfg_w = torch.tensor([0, 1], device=device)
                else:
                    cfg_w = torch.cat([torch.tensor([1], device=device),
                                       torch.linspace(0, 10, batch_size - 1, device=device)
                    ])
            else:
                if isinstance(cfg_w, list):
                    assert batch_size == len(cfg_w), f'cfg_w must have length {batch_size}, got {len(cfg_w)}'
                    cfg_w = torch.tensor(cfg_w, device=device)
                assert isinstance(cfg_w, float), f'cfg must be an float, a list of floats, or "testing", got {cfg_w}'
                cfg_w = cfg_w * torch.ones(batch_size, device=device)


        sampling_score_fn = mutils.get_score_fn(model, train=False, sampling=True, use_cfg=use_cfg, num_labels=num_labels)
        x = graph.sample_limit(*batch_dims).to(device)
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in tqdm(range(steps), desc='Sampling', disable=not use_tqdm):
            t = timesteps[i] * torch.ones(x.shape[0], device=device)
            x = projector(x)
            x = predictor.update_fn(sampling_score_fn, x, t, dt, input_label, cfg_w, use_cfg)


        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], device=device)
            x = denoiser.update_fn(sampling_score_fn, x, t, input_label, cfg_w, use_cfg)
            
        return x, input_label, cfg_w
    
    return pc_sampler

