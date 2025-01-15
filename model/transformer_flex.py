import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import torch.nn.functional as F

from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
)
from functools import lru_cache, partial
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from omegaconf import OmegaConf

from . import rotary
from .fused_add_dropout_scale import (
    bias_dropout_add_scale_fused_train, 
    bias_dropout_add_scale_fused_inference, 
    get_bias_dropout_add_scale, 
    # modulate_fused,
)

# Flex attention functions

# Define the modulate function
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if x.is_nested:
        # Convert shift and scale to nested tensors
        shift_nested = torch.nested.nested_tensor([sh.expand_as(t) for t, sh in zip(x.unbind(), shift.unbind())])
        scale_nested = torch.nested.nested_tensor([sc.expand_as(t) for t, sc in zip(x.unbind(), scale.unbind())])
        
        # Apply modulation using broadcasting
        print("Nested")
        print(x.shape)
        print(scale_nested.shape)
        print(shift_nested.shape)

        return x * (1 + scale_nested) + shift_nested
    else:
        raise NotImplementedError("I sould make sure this still works")
        return x * (1 + scale) + shift

def modulate_fused(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return modulate(x, shift, scale)

@lru_cache
def create_block_mask_cached(mask_mod, B, H, M, N, device="cuda"):
    block_mask = create_block_mask(mask_mod, B, H, M, N, device=device)
    return block_mask

def build_seq_idx(tensor: torch.Tensor):
    offsets = tensor.offsets()
    total_length = tensor.offsets()[-1].item()
    # Create a range tensor from 0 to total_length
    range_tensor = torch.arange(total_length, device="cuda", dtype=torch.int32)

    # Use searchsorted to find the index for each position
    seq_idx = torch.searchsorted(offsets, range_tensor, right=True) - 1

    return seq_idx, total_length

def create_njt_wrapper(seq_idx):
    """Generic Wrapper that makes a NJT mask_mod"""

    def njt_mask_mod(b, h, q_idx, kv_idx):
        is_same_sequence = seq_idx[q_idx] == seq_idx[kv_idx]
        return is_same_sequence

    return njt_mask_mod


#################################################################################
#                                  Layers                                       #
#################################################################################
# class LayerNorm(nn.Module):
#     def __init__(self, dim):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones([dim]))
#         self.dim = dim
#     def forward(self, x):
#         with torch.amp.autocast('cuda', enabled=False):
#             x = F.layer_norm(x.float(), [self.dim])
#         return x * self.weight[None,None,:]


def residual_linear(x, W, x_skip, residual_scale):
    """x_skip + residual_scale * W @ x"""
    dim_out, dim_in = W.shape[0], W.shape[1]
    return torch.addmm(
        x_skip.view(-1, dim_out),
        x.view(-1, dim_in),
        W.T,
        alpha=residual_scale
    ).view(*x.shape[:-1], dim_out)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256, silu=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size


    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, cond_size):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes + 1, cond_size)
        self.num_classes = num_classes

        # TODO think of initializing with 0.02 std deviation like in original DiT paper

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return embeddings
    

#################################################################################
#                                 Core Model                                    #
#################################################################################


class DDiTBlock(nn.Module):

    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.attention_head_dim = dim // n_heads

        self.norm1 = nn.LayerNorm([dim])
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm([dim])
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_ratio * dim, dim, bias=True)
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dropout = dropout
        

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )


    def forward(self, x, rotary_cos_sin, c):
        # batch_size, seq_len = x.shape[0], x.shape[1]

        bias_dropout_scale_fn = self._get_bias_dropout_scale()
        ada_out = self.adaLN_modulation(c)
        print(ada_out.shape)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = ada_out.chunk(6, dim=-1)

        # attention operation
        print("X1")
        print(x.shape)
        x_skip = x
        x = modulate_fused(self.norm1(x), shift_msa, scale_msa)
        # dtype0 = x.dtype

        print("X2")
        print(x.shape)
        qkv = self.attn_qkv(x)
        qkv = rearrange(qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        with torch.amp.autocast('cuda', enabled=False):
            cos, sin = rotary_cos_sin
            qkv = rotary.apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype)
            )
        query, key, value = rearrange(qkv, 'b s (h d) -> b s h d', h=self.n_heads).chunk(3, dim=-1)
        
        print("Query2")
        print(query.shape)
        
        # Build the seq_idx lookup table, and the block mask
        seq_idx, total_length = build_seq_idx(query)
        mask_mod_njt = create_njt_wrapper(seq_idx)

        block_mask = create_block_mask_cached(
            mask_mod_njt, 1, 1, total_length, total_length, device=query.device
        )
        x = flex_attention(
                query.view(1, -1, self.n_heads, self.attention_head_dim).transpose(1, 2),
                key.view(1, -1, self.n_heads, self.attention_head_dim).transpose(1, 2),
                value.view(1, -1, self.n_heads, self.attention_head_dim).transpose(1, 2),
                block_mask=block_mask,
        ) # returns (1, n_heads, seq_len, attention_head_dim)
        x = rearrange(x, '1 n s d -> 1 s (n d)')
        # x = rearrange(x, '(b s) h d -> b s (h d)', b=batch_size)

        x = bias_dropout_scale_fn(self.attn_out(x), None, gate_msa, x_skip, self.dropout)

        # mlp operation
        x = bias_dropout_scale_fn(self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)), None, gate_mlp, x, self.dropout)
        return x



class EmbeddingLayer(nn.Module):
    def __init__(self, dim, vocab_dim):
        """
        Mode arg: 0 -> use a learned layer, 1 -> use eigenvectors, 
        2-> add in eigenvectors, 3 -> use pretrained embedding matrix
        """
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((vocab_dim, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=math.sqrt(5))

    def forward(self, x):
        if x.is_nested:
            # Handle nested tensors
            return torch.nested.nested_tensor([self.embedding[t] for t in x.unbind()], layout=torch.jagged)
        else:
            # Handle regular tensors
            return self.embedding[x]


class DDitFinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels, cond_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm([hidden_size])
        self.linear = nn.Linear(hidden_size, out_channels)
        self.linear.weight.data.zero_()
        self.linear.bias.data.zero_()

        self.adaLN_modulation = nn.Linear(cond_dim, 2 * hidden_size, bias=True)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()


    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c)[:, None].chunk(2, dim=2)
        x = modulate_fused(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SEDD_flex(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config):
        super().__init__()

        # hack to make loading in configs easier
        if type(config) == dict:
            config = OmegaConf.create(config)

        self.config = config

        self.absorb = config.graph.type == "absorb"
        vocab_size = config.tokens + (1 if self.absorb else 0)
        self.cond_dim = config.model.cond_dim

        self.vocab_embed = EmbeddingLayer(config.model.hidden_size, vocab_size)
        self.sigma_map = TimestepEmbedder(self.cond_dim)

        self.blocks = nn.ModuleList([
            DDiTBlock(config.model.hidden_size, config.model.n_heads, config.model.cond_dim, dropout=config.model.dropout) for _ in range(config.model.n_blocks)
        ])
        assert config.model.hidden_size % config.model.n_heads == 0, "Hidden size must be divisible by number of heads."
        self.attention_head_dim = config.model.hidden_size // config.model.n_heads
        self.rotary_emb = rotary.Rotary(config.model.hidden_size // config.model.n_heads)

        self.output_layer = DDitFinalLayer(config.model.hidden_size, vocab_size, config.model.cond_dim)
        self.scale_by_sigma = config.model.scale_by_sigma

    
    def _get_bias_dropout_scale(self):
        return (
            bias_dropout_add_scale_fused_train
            if self.training
            else bias_dropout_add_scale_fused_inference
        )


    def forward(self, indices, sigma):
        for t in indices.unbind():
            print("Indices")
            print(t.shape)
        x = self.vocab_embed(indices)
        for x1 in x.unbind():
            print("Embedding")
            print(x1.shape)
        c = F.silu(self.sigma_map(sigma))

        rotary_cos_sin = self.rotary_emb(x)
        print(rotary_cos_sin.shape)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            for i in range(len(self.blocks)):
                x = self.blocks[i](x, rotary_cos_sin, c)

            x = self.output_layer(x, c)


        if self.scale_by_sigma:
            assert self.absorb, "Haven't configured this to work."
            esigm1_log = torch.where(sigma < 0.5, torch.expm1(sigma), sigma.exp() - 1).log().to(x.dtype)[:, None, None]
            x = x - esigm1_log - np.log(x.shape[-1] - 1)# this will be approximately averaged at 0
            
        x = torch.scatter(x, -1, indices[..., None], torch.zeros_like(x[..., :1]))

        return x