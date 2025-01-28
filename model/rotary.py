import torch
from torch import nn

# ## nested tensor utils

# def nested_lens(x):
#     assert x.is_nested
#     return x.offsets().diff()

# def nested_max_len(x):
#     assert x.is_nested
#     return x.offsets().diff().max()

class Rotary(torch.nn.Module):
    def __init__(self, dim, base = 10_000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        if x.is_nested:
            seq_len = x.offsets().diff().max().item()
        else:
            seq_len = x.shape[seq_dim]
            
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
            emb = torch.cat((freqs, freqs), dim=-1).to(device)
            # dims are: batch, seq_len, qkv, head, dim
            self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
            self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
            # This makes the transformation on v an identity.
            self.cos_cached[:,:,2,:,:].fill_(1.)
            self.sin_cached[:,:,2,:,:].fill_(0.)
        
        if x.is_nested:
            cos = self.cos_cached.squeeze(0)
            sin = self.sin_cached.squeeze(0)
            nested_cos_list = [cos[:n] for n in x.offsets().diff()]
            nested_sin_list = [sin[:n] for n in x.offsets().diff()]
            nested_sin = torch.nested.as_nested_tensor(nested_sin_list, device=x.device, layout=torch.jagged)
            nested_cos = torch.nested.as_nested_tensor(nested_cos_list, device=x.device, layout=torch.jagged)
            return nested_cos, nested_sin

        return self.cos_cached, self.sin_cached

    def forward(self, x, offsets=None):
        if offsets is not None:
            seq_lens = offsets.diff().tolist()
            cos_sin_list = [self.get_cos_sin(x.device, seq_len) for seq_len in seq_lens]
        else:
            cos_sin_list = self.get_cos_sin(x.device, x.shape[1])
        # if seq_len != self.seq_len_cached:
        #     self.seq_len_cached = seq_len
        #     t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        #     freqs = torch.einsum("i,j->ij", t, self.inv_freq.clone())
        #     emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
        #     # dims are: batch, seq_len, qkv, head, dim
        #     self.cos_cached = emb.cos()[None, :, None, None, :].repeat(1,1,3,1,1)
        #     self.sin_cached = emb.sin()[None, :, None, None, :].repeat(1,1,3,1,1)
        #     # This makes the transformation on v an identity.
        #     self.cos_cached[:,:,2,:,:].fill_(1.)
        #     self.sin_cached[:,:,2,:,:].fill_(0.)

        return cos_sin_list


def rotate_half(x):
    # x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :] # old implementation
    x1, x2 = x.chunk(2, dim= -1)

    return torch.cat(
        (-x2, x1), dim=-1
    )


# @torch.jit.script # TODO: I don't think this is supported for torchscript with nested tensors
# def _apply_rotary_pos_emb_torchscript(qkv, cos, sin):
def _apply_rotary_pos_emb(qkv, cos, sin): # qkv shape: (B, j1, 3, n_heads, head_dim), cos & sin shape: (1, j1.max(), 1, head_dim)

    # if qkv.is_nested:

    #     cos = cos.squeeze(0)
    #     sin = sin.squeeze(0)

    #     # slow list comprehension TODO: optimize
    #     result_list = [(t * cos[:t.shape[0]]) + (rotate_half(t) * sin[:t.shape[0]]) for t in qkv.unbind()]
         
    #     # Reassemble the list of tensors back into a nested tensor
    #     return torch.nested.as_nested_tensor(result_list, layout=torch.jagged)

    return (qkv * cos) + (rotate_half(qkv) * sin)


def apply_rotary_pos_emb(qkv, cos, sin):
    # try:
    #     import flash_attn.layers.rotary
    #     cos = cos[0,:,0,0,:cos.shape[-1]//2]
    #     sin = sin[0,:,0,0,:sin.shape[-1]//2]
    #     return flash_attn.layers.rotary.apply_rotary_emb_qkv_(
    #         qkv, cos, sin
    #     )
    # except:
    #     return _apply_rotary_pos_emb_torchscript(qkv, cos, sin)
    return _apply_rotary_pos_emb(qkv, cos, sin)