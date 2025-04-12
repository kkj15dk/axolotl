import abc
import torch
import torch.nn as nn
import numpy as np


def get_noise(type, sigma_min, sigma_max, eps):
    if type == "geometric":
        return GeometricScheduler(sigma_min, sigma_max)
    elif type == "cosine":
        return CosineScheduler(eps)
    elif type == "linear":
        return LinearScheduler(eps)
    elif type == "loglinear":
        return LogLinearNoise(eps)
    else:
        raise ValueError(f"{type} is not a valid noise")

class Noise(abc.ABC, nn.Module):
    """
    Baseline forward method to get the total + rate of noise at a timestep
    """
    def forward(self, t):
        return self.total_noise(t), self.rate_noise(t)

    """
    Assume time goes from 0 to 1
    """
    @abc.abstractmethod
    def rate_noise(self, t):
        """
        Rate of change of noise ie g(t)
        """
        pass

    @abc.abstractmethod
    def total_noise(self, t):
        """
        Total noise ie \int_0^t g(t) dt + g(0)
        """
        pass


class LogLinearNoise(Noise, nn.Module):
    """
    Log Linear noise schedule built so that 1 - 1/e^(n(t)) interpolates between 0 and ~1
    when t goes from 0 to 1. Used for absorbing

    Total noise is -log(1 - (1 - eps) * t), so the sigma will be (1 - eps) * t
    """
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps
        self.empty = nn.Parameter(torch.tensor(0.0))

    def rate_noise(self, t):
        return (1 - self.eps) / (1 - (1 - self.eps) * t)

    def total_noise(self, t):
        return -torch.log1p(-(1 - self.eps) * t)


class Scheduler(abc.ABC, nn.Module):
    """
    Baseline forward method to get alphas and betas for the MD4 implementation of discrete diffusion (https://arxiv.org/pdf/2406.04329)
    """

    @abc.abstractmethod
    def __init__(self, eps):
        super().__init__()
        self.eps = eps

    def forward(self, t, beta=False, dbeta=False, alpha=False, dalpha=False, dgamma_times_alpha=False): # TODO: this should probably be implemented differently, if MD4 is better than SEDD
        """
        return total noise, rate noise, to be compatible with SEDD loss function
        """
        
        output = ()

        if beta:
            output = output + (self.beta(t),)
        if dbeta:
            output = output + (self.dbeta(t),)
        if alpha:
            output = output + (self.alpha(t),)
        if dalpha:
            output = output + (self.dalpha(t),)
        if dgamma_times_alpha:
            output = output + (self.dgamma_times_beta(t),)

        if len(output) == 1:
            return output[0]

        return output

    def beta(self, t):
        return -self.alpha(t).log()
    
    def dbeta(self, t):
        return - self.dalpha(t) / self.alpha(t)

    def alpha(self, t):
        return (1.0 - 2 * self.eps) * self._alpha(t) + self.eps
    
    def dalpha(self, t):
        return (1.0 - 2 * self.eps) * self._dalpha(t)

    def dgamma_times_beta(self, t):
        return - self.dalpha(t) / (1 - self.alpha(t))
    

    @abc.abstractmethod
    def _alpha(self, t):
        pass

    @abc.abstractmethod
    def _dalpha(self, t):
        pass


class CosineScheduler(Scheduler, nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__(eps)
        self.empty = nn.Parameter(torch.tensor(0.0))
    
    def _dalpha(self, t):
        return -torch.pi / 2 * torch.cos(torch.pi / 2 * t)
    
    def _alpha(self, t):
        return 1 - torch.cos(torch.pi / 2 * (1 - t))


class LinearScheduler(Scheduler, nn.Module):
    def __init__(self, eps=1e-4):
        super().__init__(eps)
        self.empty = nn.Parameter(torch.tensor(0.0))
    
    def _dalpha(self, t):
        return - torch.ones_like(t)
    
    def _alpha(self, t):
        return 1 - t


class GeometricScheduler(Scheduler, nn.Module):
    def __init__(self, sigma_min=1e-3, sigma_max=1, learnable=False):
        super().__init__()
        self.sigmas = 1.0 * torch.tensor([sigma_min, sigma_max])
        if learnable:
            self.sigmas = nn.Parameter(self.sigmas)
        self.empty = nn.Parameter(torch.tensor(0.0))

    def _dalpha(self, t):
        self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t * (self.sigmas[0].log() - self.sigmas[1].log()) * torch.exp(-1.0 * self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t)
    
    def _alpha(self, t):
        return torch.exp(-1.0 * self.sigmas[0] ** (1 - t) * self.sigmas[1] ** t)