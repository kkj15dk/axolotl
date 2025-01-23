import torch

def rotate_half(x):
    # x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :] # old implementation
    x1, x2 = x.chunk(2, dim= -1)

    return torch.cat(
        (-x2, x1), dim=-1
    )


# @torch.jit.script # TODO: I don't think this is supported for torchscript with nested tensors
# def _apply_rotary_pos_emb_torchscript(qkv, cos, sin):
def _apply_rotary_pos_emb(qkv, cos, sin): # qkv shape: (B, j1, 3, n_heads, head_dim), cos & sin shape: (1, j1.max(), 1, head_dim)

    return (qkv * cos) + (rotate_half(qkv) * sin)


def test_apply_rotary_pos_emb_non_nested():
    qkv = torch.randn(2, 64, 3, 8, 16)
    cos = torch.randn(1, 64, 3, 1, 16)
    sin = torch.randn(1, 64, 3, 1, 16)
    
    result = _apply_rotary_pos_emb(qkv, cos, sin)
    
    print("Non-nested tensor result shape:", result.shape)
    assert result.shape == qkv.shape
    assert not result.is_nested

def test_apply_rotary_pos_emb_nested():
    qkv = torch.nested.nested_tensor([torch.randn(64, 3, 8, 16), torch.randn(32, 3, 8, 16)], requires_grad=False, layout=torch.jagged)
    cos = torch.randn(1, 64, 3, 1, 16)
    sin = torch.randn(1, 64, 3, 1, 16)
    
    result = _apply_rotary_pos_emb(qkv, cos, sin)
    
    print("Nested tensor result shapes:", [t.shape for t in result.unbind()])
    assert result.is_nested
    assert result[0].shape == qkv[0].shape
    assert result[1].shape == qkv[1].shape

if __name__ == '__main__':
    test_apply_rotary_pos_emb_non_nested()
    test_apply_rotary_pos_emb_nested()
    print("All tests passed.")