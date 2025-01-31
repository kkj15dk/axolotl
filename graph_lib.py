import abc
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from model.nested_utils import (
    expand_using_offsets,
    flatten_nested_tensor,
)

from catsample import sample_categorical

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
    def score_entropy(self, score, sigma, x, x0, offsets=None):
        """
        Computes the score entropy function (with requisite constant normalization)
        """
        pass


class Uniform(Graph):
    """
    Everything goes to everything else. Normalized down by dimension to avoid blowup.
    """
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim
    
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
        trans: torch.Tensor = torch.ones(*i.shape, self.dim, device=i.device) * (1 - (-sigma[..., None]).exp()) / self.dim
        trans = trans.scatter(-1, i[..., None], torch.zeros_like(trans))
        trans = trans.scatter(-1, i[..., None], 1 - trans.sum(dim=-1, keepdim=True))
        return trans
    
    def transp_transition(self, i, sigma):
        return self.transition(i, sigma)

    def sample_transition(self, i, sigma):
        move_chance = 1 - (-sigma).exp()
        if i.is_nested:
            raise NotImplementedError("Nested tensors not supported yet")
        move_indices = torch.rand(*i.shape, device=i.device) < move_chance
        i_pert = torch.where(move_indices, torch.randint_like(i, self.dim), i)
        return i_pert

    def staggered_score(self, score, dsigma):
        dim = score.shape[-1]
        epow = (-dsigma).exp()[..., None]
        return ((epow - 1) / (dim * epow)) * score.sum(dim=-1, keepdim=True) + score / epow

    def sample_limit(self, *batch_dims):
        return torch.randint(0, self.dim, batch_dims)

    def score_entropy(self, score, sigma, x, x0, offsets=None):
        esigm1 = torch.where( # more precise, compute differently for sigma under 0.5, than for over 0.5. (exp(sigma) - 1)
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        ratio = 1 - self.dim / (esigm1 + self.dim)

        # negative term
        neg_term = score.mean(dim=-1) - torch.gather(score, -1, x[..., None]).squeeze(-1) / self.dim
        # no move means scaling by the uniform ratio. move means alter only one ratio away from 1
        neg_term = torch.where(
            x == x0,
            ratio * neg_term,
            torch.gather(score, -1, x0[..., None]).squeeze(-1) / esigm1 + neg_term
        )

        # constant factor
        const = torch.where(
            x == x0,
            (self.dim - 1) / self.dim * ratio * (ratio.log() - 1),
            ((-ratio.log() - 1) / ratio - (self.dim - 2)) / self.dim 
        )

        #positive term
        sexp = score.exp()
        pos_term = sexp.mean(dim=-1) - torch.gather(sexp, -1, x[..., None]).squeeze(-1) / self.dim
        return pos_term - neg_term + const


class Absorbing(Graph): #  we exponentiate (which maintains positivity) to be beneficial to avoid numerical errors and also found that scaling by e^σ − 1 helps for absorbing diffusion
    def __init__(self, dim):
        super().__init__()
        self._dim = dim

    @property
    def dim(self):
        return self._dim + 1
    
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

    def sample_transition(self, i: torch.Tensor, sigma):
        move_chance = 1 - (-sigma).exp()
        # print("move_chance", move_chance)
        if i.is_nested:
            # print("Nested")
            # # Handle nested tensors
            # for t in i.unbind():
            #     print("i", t.shape)
            move_indices_list = [
                torch.rand(len, device=i.device) < mc.item() for len, mc in zip(i.offsets().diff(), move_chance.unbind())
            ]
            # print("move_indices_list", move_indices_list)
            move_indices = torch.nested.nested_tensor(
                move_indices_list, layout=torch.jagged
            )
            # print("move_indices", move_indices)
            # for t in i.unbind():
            #     print("i", t.shape)
            # for t in move_indices.unbind():
            #     print("move_indices", t.shape)

            # TODO: for some reason masked_fill does not work, even though i and move_indices are the same shape
            # i_pert = i.masked_fill(move_indices, self.dim - 1)
            # Usng where instead

            i_pert = torch.nested.nested_tensor([
                torch.where(mi, self.dim - 1, t) for mi, t in zip(move_indices.unbind(), i.unbind())
            ], layout=torch.jagged)
            
            # print("i_pert", i_pert)
            # for t in i_pert.unbind():
            #     print("i_pert", t.shape)
        else:
            # Handle regular tensors
            move_indices = torch.rand(*i.shape, device=i.device) < move_chance
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

    def score_entropy(self, score, sigma, x, x0, offsets=None):
        esigm1 = torch.where( # more precise, compute differently for sigma under 0.5, than for over 0.5. (exp(sigma) - 1)
            sigma < 0.5,
            torch.expm1(sigma),
            torch.exp(sigma) - 1
        )
        # print("esigm1", esigm1.shape)
        if offsets is not None:
            esigm1 = expand_using_offsets(esigm1, offsets=offsets).squeeze(-1) # (B, 1) -> (BL)
            # print("esigm1", esigm1.shape)
            # print("x", x.shape)
            x = flatten_nested_tensor(x) # (B, L) -> (BL)
            # print("x", x.shape)
            # print("x0", x0.shape)
            x0 = flatten_nested_tensor(x0)  # (B, L) -> (BL)
            # print("x0", x0.shape)
        else:
            esigm1 = esigm1.expand_as(x)

        rel_ind = x == self.dim - 1 # Where x is the absorbing state (mask)
        # print("rel_ind", rel_ind.shape)
        # print("score", score.shape)

        ratio = 1 / esigm1[rel_ind]
        # print("ratio", ratio.shape)
        other_ind = x0[rel_ind] # The state we want to go to (target)
        # print("other_ind", other_ind.shape)

        # negative_term
        neg_term = ratio * torch.gather(score[rel_ind], -1, other_ind[..., None]).squeeze(-1)
        # print("neg_term", neg_term.shape)

        # positive term
        pos_term = score[rel_ind][:, :-1].exp().sum(dim=-1)
        # print("pos_term", pos_term.shape)

        # constant term
        const = ratio * (ratio.log() - 1)
        # print("const", const.shape)

        entropy = torch.zeros(*x.shape, device=x.device)
        # print("entropy", entropy.shape)
        entropy[rel_ind] += pos_term - neg_term + const
        print("entropy", entropy)

        return entropy