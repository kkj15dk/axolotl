import abc
import torch
import torch.nn.functional as F
from .catsample import sample_categorical

from .model import utils as mutils
from typing import Optional, Union
from tqdm import tqdm
from typing import List

from .noise_lib import Scheduler
from .graph_lib import Graph

from wandb import Table

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


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, graph: Graph, noise: Scheduler):
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


def get_predictor(name) -> Predictor:
    return _PREDICTORS[name]


def write_samples(output: str, 
                  sequences: List[str], 
                  sampling_label: torch.LongTensor, 
                  sampling_cfg_w: torch.FloatTensor, 
                  steps: int, 
                  name: str = "sample",
                  use_wandb: bool = False,
                  current_table: Optional[Table] = None,
                  mode: str = 'a'
):
    """Write samples to a file."""
    
    assert sampling_cfg_w.dim() == 1, f"cfg_w must be a 1D tensor, got {sampling_cfg_w.dim()}"
    assert len(sequences) == len(sampling_label) == len(sampling_cfg_w), f"Length mismatch: {len(sequences)}, {len(sampling_label)}, {len(sampling_cfg_w)}"
    if use_wandb:
        assert current_table is not None, "current_table must be provided if use_wandb is True"

    print(f"Writing samples to {output}")
    with open(output, mode) as file:
        for i, seq in enumerate(sequences):
            if sampling_label[i] == 0:
                sequence_label = "prokaryotic"
            elif sampling_label[i] == 1:
                sequence_label = "eukaryotic"
            elif sampling_label[i] == 2:
                sequence_label = "none"
            else:
                raise ValueError(f"Invalid label: {sampling_label[i]}")
            w = sampling_cfg_w[i].item()
            
            file.write(f">{name}_{i} label:{sequence_label} cfg_w:{w} steps:{steps}\n")
            file.write(seq + "\n")

            if use_wandb:
                current_table.add_data(steps, i, sequence_label, w, steps, seq) # workaround


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


@register_predictor(name="none")
class NonePredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, label, cfg_w):
        return x


@register_predictor(name="euler_score")
class EulerPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, label, cfg_w, use_cfg=False):
        sigma, dsigma = self.noise(t, beta=True, dbeta=True)

        score = score_fn(x, t, label, sigma)
        print("score", score[0])

        if use_cfg:
            score = classifier_free_guidance(score, cfg_w)

        dsigma = dsigma.unsqueeze(-1) # TODO: make it so this is not necessary
        rev_rate = step_size * dsigma[..., None] * self.graph.reverse_rate(x, score)
        print("rev_rate", rev_rate[0])
        x = self.graph.sample_rate(x, rev_rate)
        print(x[0])
        return x


@register_predictor(name="analytic_score")
class AnalyticPredictor(Predictor):
    def update_fn(self, score_fn, x, t, step_size, label, cfg_w, use_cfg=False):
        curr_beta = self.noise(t, beta=True)
        next_beta = self.noise(t - step_size, beta=True)
        dsigma = curr_beta - next_beta

        score = score_fn(x, t, label, curr_beta)

        if use_cfg:
            score = classifier_free_guidance(score, cfg_w)

        dsigma = dsigma.unsqueeze(-1) # TODO: make it so this is not necessary
        stag_score = self.graph.staggered_score(score, dsigma)
        probs = stag_score * self.graph.transp_transition(x, dsigma)
        return sample_categorical(probs)


@register_predictor(name="ancestral_x0")
class AncestralPredictor(Predictor):
    def update_fn(self, x0_fn, x, t, step_size, label, cfg_w, use_cfg=False):
        alpha_t = self.noise(t, alpha=True)
        alpha_s = self.noise(t - step_size, alpha=True)
        unmask_prob = ((alpha_s - alpha_t) / (1 - alpha_t)).unsqueeze(-1).unsqueeze(-1)

        x0_prediction = x0_fn(x, t, label)

        if use_cfg:
            x0_prediction = classifier_free_guidance(x0_prediction, cfg_w)

        if self.graph.absorb:
            masking_state = F.one_hot(self.graph.vocab_size * torch.ones_like(x, dtype=torch.long, device=x0_prediction.device), num_classes=self.graph.dim) # one-hot encoding of the absorbing state
            probs = unmask_prob * x0_prediction + (1 - unmask_prob) * masking_state 
        else:
            raise NotImplementedError("Not implemented yet")

        return sample_categorical(probs)


class Denoiser:
    def __init__(self, graph, noise, prediction_type):
        self.graph: Graph = graph
        self.noise = noise
        self.prediction_type = prediction_type

    def update_fn(self, output_fn, x, t, label, cfg_w, use_cfg=False):
        beta = self.noise(t, beta=True)

        if self.prediction_type == 'log_score':
            output = output_fn(x, t, label, beta)
        elif self.prediction_type == 'x0':
            output = output_fn(x, t, label)
        else:
            raise ValueError(f"Invalid prediction type: {self.prediction_type}")

        if use_cfg:
            output = classifier_free_guidance(output, cfg_w)

        if self.prediction_type == 'log_score':
            beta = beta.unsqueeze(-1) # TODO: make it so this is not necessary
            stag_score = self.graph.staggered_score(output, beta) # beta = dbeta for the last step
            probs = stag_score * self.graph.transp_transition(x, beta)
        elif self.prediction_type == 'x0':
            probs = output
        else:
            raise ValueError(f"Invalid prediction type: {self.prediction_type}")

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
                                 use_tqdm=False,
                                 prediction_type=config.prediction_type
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
                   use_tqdm: bool=False,
                   prediction_type: str=None,
                   print_intermediates: bool=True,
):
    if prediction_type == 'x0':
        assert predictor == 'ancestral_x0', "Prediction type x0 requires the predictor to be ancestral_x0"
    elif prediction_type == 'log_score':
        assert predictor in ['euler_score', 'analytic_score'], "Prediction type log_score requires the predictor to be euler_score or analytic_score"
    else:
        raise ValueError(f"Invalid prediction type: {prediction_type}")
    
    predictor: Predictor = get_predictor(predictor)(graph, noise)
    projector = proj_fun
    denoiser = Denoiser(graph, noise, prediction_type)

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
            use_cfg = False # No need for interpolation at cfg_w = 0
            cfg_w = torch.zeros(batch_size, device=device)
        elif cfg_w == 1: # conditional sampling
            use_cfg = False # No need for interpolation at cfg_w = 1
            cfg_w = torch.ones(batch_size, device=device)

        else: # We are interpolating or extrapolating
            use_cfg = True
            if cfg_w == 'testing':
                if batch_size == 1:
                    cfg_w = torch.tensor([1], device=device)
                elif batch_size == 2:
                    cfg_w = torch.tensor([1, 0], device=device)
                else:
                    cfg_w = torch.cat([torch.tensor([1], device=device),
                                       torch.linspace(0, 5, batch_size - 1, device=device)
                    ])
            else:
                if isinstance(cfg_w, list):
                    assert batch_size == len(cfg_w), f'cfg weight must have length {batch_size}, got {len(cfg_w)}'
                    cfg_w = torch.tensor(cfg_w, device=device)
                else:
                    assert isinstance(cfg_w, float), f'cfg weight must be a float, a list of floats, or "testing", got {cfg_w}'
                    cfg_w = cfg_w * torch.ones(batch_size, device=device)

        if prediction_type == 'x0':
            sampling_output_fn = mutils.get_output_fn(model, train=False, exponentiate=False, use_cfg=use_cfg, num_labels=num_labels)
        elif prediction_type == 'log_score':
            sampling_output_fn = mutils.get_output_fn(model, train=False, exponentiate=True, use_cfg=use_cfg, num_labels=num_labels)
        else:
            raise ValueError(f"Invalid prediction type: {prediction_type}")
        
        x = graph.sample_limit(*batch_dims).to(device)
        if print_intermediates:
                print(x[0])
        timesteps = torch.linspace(1, eps, steps + 1, device=device)
        dt = (1 - eps) / steps

        for i in tqdm(range(steps), desc='Sampling', disable=not use_tqdm):
            t = timesteps[i] * torch.ones(x.shape[0], device=device)
            x = projector(x)
            x = predictor.update_fn(sampling_output_fn, x, t, dt, input_label, cfg_w, use_cfg)
            if print_intermediates:
                print(x[0])
            
            if i > 5:
                raise ValueError("Too many steps")


        if denoise:
            # denoising step
            x = projector(x)
            t = timesteps[-1] * torch.ones(x.shape[0], device=device)
            x = denoiser.update_fn(sampling_output_fn, x, t, input_label, cfg_w, use_cfg)
            if print_intermediates:
                print(x[0])
            
        return x, input_label, cfg_w
    
    return pc_sampler

