import torch
import torch.nn as nn
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.functional import scaled_dot_product_attention
from einops import rearrange

BATCH = 4 
EMB_DIM = 256 
HEADS = 8 
TOKENS = 512

batch = torch.nested.nested_tensor([torch.rand(TOKENS  // (i+1), EMB_DIM * HEADS) for i in range(BATCH)], dtype=torch.float, device="cuda", layout = torch.jagged)


def trace_ready(p):
    print('trace ready!')
    import os
    os.system('rm chrome_trace.gz')
    p.export_chrome_trace('chrome_trace')
    os.system('gzip chrome_trace')

class test_model(nn.Module):
    def __init__(
            self,
            heads,
            dim,
    ):
        super().__init__()
        self.head_dim = dim
        self.hidden_dim = dim * heads
        self.heads = heads
        self.qkv_proj = nn.Linear(self.hidden_dim, self.hidden_dim * 3)

    def forward(self, x: torch.Tensor):

        qkv: torch.Tensor = self.qkv_proj(x) # (b, l, d*h*3)
        q,k,v = qkv.chunk(3, dim=-1)
        q = q.unflatten(-1, [self.heads, self.head_dim]).transpose(1, 2)
        k = k.unflatten(-1, [self.heads, self.head_dim]).transpose(1, 2)
        v = v.unflatten(-1, [self.heads, self.head_dim]).transpose(1, 2)

        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],on_trace_ready=trace_ready,with_stack=True):
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                out = scaled_dot_product_attention(
                    q, k, v, attn_mask=None, dropout_p=0.0
                )
        return out

model = test_model(HEADS, EMB_DIM).to('cuda')

model.train()

out = model(batch)

print(out)