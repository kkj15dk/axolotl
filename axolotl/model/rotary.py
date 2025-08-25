import torch
from torch import nn
import time

# ## nested tensor utils

# def nested_lens(x):
#     assert x.is_nested
#     return x.offsets().diff()

# def nested_max_len(x):
#     assert x.is_nested
#     return x.offsets().diff().max()

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)) # (dim / 2), dim must be even
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1, max_len=None):
        if x.is_nested:
            if max_len is not None:
                seq_len = max_len
            else:
                seq_len = x.offsets().diff().max().item()
        else:
            seq_len = x.shape[seq_dim]
            
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
            # This makes the transformation on v an identity.
            self.cos_cached[:,:,2,:,:].fill_(1.)
            self.sin_cached[:,:,2,:,:].fill_(0.)
        
        if x.is_nested: # TODO: I think this can be rewritten to avoid the nested tensor ops, because it is relative encoding. The start-index should not matter, and maybe we can just apply to the entire value tensor.

            # Get the lengths of the nested tensor, also implicitly get the batch size
            lengths = x.offsets().diff()
            batch_size = len(lengths)

            # Expand the cached tensors to match the batch size
            cos_nested = self.cos_cached.repeat(batch_size, 1, 1, 1, 1)
            sin_nested = self.sin_cached.repeat(batch_size, 1, 1, 1, 1)

            # Take the nested view of the dense tensors
            cos_nested = torch.nested.narrow(
                cos_nested, 
                dim=1,
                start=0,
                length=lengths,
                layout=torch.jagged
            ).contiguous()
            sin_nested = torch.nested.narrow(
                sin_nested, 
                dim=1,
                start=0,
                length=lengths,
                layout=torch.jagged
            ).contiguous()

            return cos_nested, sin_nested

        return self.cos_cached, self.sin_cached


def rotate_half(x):
    # x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :] # old implementation
    x1, x2 = x.chunk(2, dim= -1)

    return torch.cat(
        (-x2, x1), dim=-1
    )


# @torch.jit.script # TODO: I don't think this is supported for torchscript with nested tensors
# def _apply_rotary_pos_emb_torchscript(qkv, cos, sin):
def _apply_rotary_pos_emb(qkv, cos, sin): 

    return (qkv * cos) + (rotate_half(qkv) * sin)


def apply_rotary_pos_emb(qkv, cos, sin):

    return _apply_rotary_pos_emb(qkv, cos, sin)