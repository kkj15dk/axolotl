import abc
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .utils_nested import packed_tensor_from_jagged, jagged_from_packed_tensor, coerce_offsets, expand_using_offsets

from .catsample import sample_categorical

def get_graph(config, device):
    if config.graph.type == "uniform":
        return Uniform(config.tokens)
    elif config.graph.type == "absorb":
        return Absorbing(config.tokens)
    else:
        raise ValueError(f"Graph {config.graph.type} not valid")


def unsqueeze_as(x, y, back=True):
    if back:
        return x.view(*x.shape, *((1,) * (len(y.shape) - len(x.shape))))
    else:
        return x.view(*((1,) * (len(y.shape) - len(x.shape))), *x.shape)


class Graph(abc.ABC):

    @property
    def dim(self):
        pass

    @property
    def vocab_size(self):
        pass

    @property
    def absorb(self):
        """
        Whether input {dim - 1} is an absorbing state (used for denoising to always remove the mask).
        """
        pass


    @abc.abstractmethod
    def rate(self, i):
        """
        Computes the i-th column of the rate matrix Q, where i is [B_1, ..., B_n].

        This is intended to compute the "forward" rate of p(X_t | X_0 = i).
        """
        pass


    @abc.abstractmethod
    def transp_rate(self, i):
        """
        Computes the i-th row of the rate matrix Q.

        Can be used to compute the reverse rate.
        """
        pass


    @abc.abstractmethod
    def transition(self, i, sigma):
        """
        Computes the i-th column of the transition matrix e^{sigma Q}.
        """
        pass


    @abc.abstractmethod
    def transp_transition(self, i, sigma):
        pass
    

    def sample_transition(self, i, sigma):
        """
        Samples the transition vector.
        """
        transition_vector = self.transition(i, sigma)
        return sample_categorical(transition_vector, method="hard")
    

    def reverse_rate(self, i, score):
        """
        Constructs the reverse rate. Which is score * transp_rate
        """
        normalized_rate = self.transp_rate(i) * score

        normalized_rate.scatter_(-1, i[..., None], torch.zeros_like(normalized_rate))
        normalized_rate.scatter_(-1, i[..., None], -normalized_rate.sum(dim=-1, keepdim=True))
        return normalized_rate


    def sample_rate(self, i, rate):
        return sample_categorical(F.one_hot(i, num_classes=self.dim).to(rate) + rate)

    
    @abc.abstractmethod
    def staggered_score(self, score, dsigma):
        """
        Computes p_{sigma - dsigma}(z) / p_{sigma}(x), which is approximated with
        e^{-{dsigma} E} score
        """
        pass
    

    @abc.abstractmethod
    def sample_limit(self, *batch_dims):
        """
        Sample the limiting distribution. Returns the probability vector as well.
        """
        pass


    @abc.abstractmethod
    def score_entropy(self, log_score, beta, x, x0):
        """
        Computes the score entropy function (with requisite constant normalization)
        """
        pass

    @abc.abstractmethod
    def x0_entropy(self, logits, alpha_t1, dgamma_times_alpha, x, x0):
        """
        Computes the x0 prediction loss function
        """
        pass


class Uniform(Graph):
    """
    Everything goes to everything else. Normalized down by dimension to avoid blowup.
    """
    def __init__(self, dim):
        self._vocab_size = dim

    @property
    def vocab_size(self):
        return self._vocab_size

    @property
    def dim(self):
        return self._vocab_size
    
    @property
    def absorb(self):
        return False

    def rate(self, i):
        edge = torch.ones(*i.shape, self.dim, device=i.device) / self.dim
        edge = edge.scatter(-1, i[..., None], - (self.dim - 1) / self.dim)
        return edge

    def transp_rate(self, i):
        return self.rate(i)

    def transition(self, i, sigma):
        trans = torch.ones(*i.shape, self.dim, device=i.device) * (1 - (-sigma[..., None]).exp()) / self.dim
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        return trans
    
    def transp_transition(self, i, sigma):
        return self.transition(i, sigma)

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand_like(i.float(), device=i.device) < move_chance
        i_pert = torch.where(move_indices, torch.randint_like(i, self.dim), i)
        return i_pert

    def staggered_score(self, score, dsigma):
        dim = score.shape[-1]
        epow = (-dsigma).exp()[..., None]
        return ((epow - 1) / (dim * epow)) * score.sum(dim=-1, keepdim=True) + score / epow

    def sample_limit(self, *batch_dims):
        return torch.randint(0, self.dim, batch_dims)

    def score_entropy(self, log_score, beta, x, x0):
        esigm1 = torch.where(
            beta < 0.5,
            torch.expm1(beta),
            torch.exp(beta) - 1
        )
        if log_score.is_nested:
            log_score, offsets = packed_tensor_from_jagged(log_score)
            x, _ = packed_tensor_from_jagged(x)
            x0, _ = packed_tensor_from_jagged(x0)
            esigm1, _ = expand_using_offsets(esigm1, offsets)
        else:
            esigm1 = esigm1.expand_as(x)
            offsets = None

        ratio = 1 - self.dim / (esigm1 + self.dim)

        # negative term
        neg_term = log_score.mean(dim=-1) - torch.gather(log_score, -1, x[..., None]).squeeze(-1) / self.dim
        # no move means scaling by the uniform ratio. move means alter only one ratio away from 1
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(log_score, -1, x0[..., None]).squeeze(-1) / esigm1 + neg_term
        )

        # constant factor
        const = torch.where(
            x == x0,
            (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim 
        )

        # positive term
        sexp = log_score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / self.dim
        entropy = pos_term - neg_term + const
        
        if offsets is not None:
            entropy = jagged_from_packed_tensor(entropy, offsets) # B x j1
        
        return entropy

    def recon_loss(self, alpha_t1):
        # raise NotImplementedError("Reconstruction loss not implemented for uniform graph")
        loss_recon = (
            (1 - alpha_t1)
            * np.log(self.dim)
        ) # B
        return loss_recon
    
    def latent_loss(self):
        # negligible
        return 0

    def diffusion_loss(self, logits, dgamma_times_alpha, x, x0):

        # convert to flat tensors
        if logits.is_nested:
            logits, offsets = packed_tensor_from_jagged(logits)
            x, _ = packed_tensor_from_jagged(x)
            x0, _ = packed_tensor_from_jagged(x0)
            dgamma_times_alpha, _ = expand_using_offsets(dgamma_times_alpha, offsets)
        else:
            raise NotImplementedError("Diffusion loss not tested yet for normal, non-nested tensors")
            dgamma_times_alpha = dgamma_times_alpha.expand_as(x)
            offsets = None
        
        # MD4 implementation, translated from jax
        log_p = torch.log_softmax(logits, dim=-1)
        one_hot_x0 = F.one_hot(x0, num_classes=self.dim)
        neg_cross_entropy = torch.where(one_hot_x0.to(dtype=torch.bool), log_p, 0) # to avoid nans when with -inf * 0
        neg_cross_entropy = torch.sum(neg_cross_entropy, dim=-1)
        neg_cross_entropy = dgamma_times_alpha * neg_cross_entropy

        if offsets is not None:
            neg_cross_entropy = jagged_from_packed_tensor(neg_cross_entropy, offsets) # (B, j1)
        
        return neg_cross_entropy
    
    def x0_entropy(self, logits, alpha_t1, dgamma_times_alpha, x, x0):

        # reconsuction loss:
        loss_recon = self.recon_loss(alpha_t1) # int

        # latent loss:
        loss_prior = self.latent_loss() # 0

        # diffusion loss:
        loss_diffusion = self.diffusion_loss(logits, dgamma_times_alpha, x, x0) # (B, j1)

        loss = loss_recon + loss_prior + loss_diffusion

        return loss

class Absorbing(Graph):
    def __init__(self, dim):
        super().__init__()
        self._vocab_size = dim # vocab size without absorbing state

    @property
    def vocab_size(self):
        return self._vocab_size
    
    @property
    def dim(self):
        return self._vocab_size + 1
    
    @property
    def absorb(self):
        return True

    def rate(self, i):
        # edge = - F.one_hot(i, num_classes=self.dim)
        # edge.scatter_add_(-1, i[..., None], torch.ones_like(edge[..., :1]))
        return F.one_hot((self.dim - 1) * torch.ones_like(i), num_classes=self.dim) - F.one_hot(i, num_classes=self.dim)        

    def transp_rate(self, i):
        edge = -F.one_hot(i, num_classes=self.dim)
        edge[i == self.dim - 1] += 1
        return edge

    def transition(self, i, sigma):
        pass
    
    def transp_transition(self, i, sigma):
        sigma = unsqueeze_as(sigma, i[..., None])
        edge = (-sigma).exp() * F.one_hot(i, num_classes=self.dim)
        edge += torch.where(
            i == self.dim - 1,
            1 - (-sigma).squeeze(-1).exp(),
            0
        )[..., None]
        return edge

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        move_indices = torch.rand_like(i.float(), device=i.device) < move_chance
        i_pert = torch.where(move_indices, self.dim - 1, i)
        return i_pert
    
    def staggered_score(self, score, dsigma):
        score = score.clone() # yeah yeah whatever we should probably do this
        extra_const = (1 - (dsigma).exp()) * score.sum(dim=-1)
        score *= dsigma.exp()[:, None]
        score[..., -1] += extra_const
        return score

    def sample_limit(self, *batch_dims):
        return (self.dim - 1) * torch.ones(*batch_dims, dtype=torch.int64)

    def score_entropy(self, log_score, beta, x, x0):
        ebetam1 = torch.where(
            beta < 0.5,
            torch.expm1(beta),
            torch.exp(beta) - 1
        )

        if log_score.is_nested:
            log_score, offsets = packed_tensor_from_jagged(log_score)
            x, _ = packed_tensor_from_jagged(x)
            x0, _ = packed_tensor_from_jagged(x0)
            ebetam1, _ = expand_using_offsets(ebetam1, offsets)
        else:
            ebetam1 = ebetam1.expand_as(x)
            offsets = None
        
        rel_ind = x == self.dim - 1
        ratio = 1 / ebetam1[rel_ind]
        other_ind = x0[rel_ind]

        # negative_term
        neg_term = ratio * torch.gather(log_score[rel_ind], -1, other_ind[..., None]).squeeze(-1)

        # positive term
        pos_term = log_score[rel_ind][:, :-1].exp().sum(dim=-1)

        # constant term
        const = ratio * (ratio.log() - 1)

        entropy = torch.zeros(*x.shape, device=x.device)
        entropy[rel_ind] += pos_term - neg_term + const

        if offsets is not None:
            entropy = jagged_from_packed_tensor(entropy, offsets)

        return entropy

    def recon_loss(self, alpha_t1):
        loss_recon = (
            (1 - alpha_t1)
            * np.log(self.dim)
        ) # B
        return loss_recon
    
    def latent_loss(self):
        # negligible
        return 0

    def diffusion_loss(self, logits, dgamma_times_alpha, x, x0):

        # convert to flat tensors
        if logits.is_nested:
            logits, offsets = packed_tensor_from_jagged(logits)
            x, _ = packed_tensor_from_jagged(x)
            x0, _ = packed_tensor_from_jagged(x0)
            dgamma_times_alpha, _ = expand_using_offsets(dgamma_times_alpha, offsets)
        else:
            raise NotImplementedError("Diffusion loss not tested yet for normal, non-nested tensors")
            dgamma_times_alpha = dgamma_times_alpha.expand_as(x)
            offsets = None
        
        # MD4 implementation, translated from jax
        log_p = torch.log_softmax(logits, dim=-1)
        one_hot_x0 = F.one_hot(x0, num_classes=self.dim)
        neg_cross_entropy = torch.where(one_hot_x0.to(dtype=torch.bool), log_p, 0) # to avoid nans when with -inf * 0
        neg_cross_entropy = torch.sum(neg_cross_entropy, dim=-1)

        mask = (x == self.vocab_size)
        neg_cross_entropy = torch.where(mask, dgamma_times_alpha * neg_cross_entropy, 0) # to avoid nans when with -inf * 0

        if offsets is not None:
            neg_cross_entropy = jagged_from_packed_tensor(neg_cross_entropy, offsets) # (B, j1)
        
        return neg_cross_entropy
    
    def x0_entropy(self, logits, alpha_t1, dgamma_times_alpha, x, x0):

        # reconsuction loss:
        loss_recon = self.recon_loss(alpha_t1) # int

        # latent loss:
        loss_prior = self.latent_loss() # 0

        # diffusion loss:
        loss_diffusion = self.diffusion_loss(logits, dgamma_times_alpha, x, x0) # (B, j1)

        loss = loss_recon + loss_prior + loss_diffusion

        return loss