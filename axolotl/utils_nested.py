import torch

def packed_tensor_from_jagged(tensor):
    assert tensor.is_nested, "Input tensor must be nested"
    return tensor.values(), tensor.offsets()

def jagged_from_packed_tensor(tensor, offsets):
    '''
    Make a jagged tensor from values and offsets. Preserves gradient.
    '''
    lens: torch.Tensor = offsets.diff()
    min_len, max_len = lens.aminmax()

    return torch.nested.nested_tensor_from_jagged(
        values=tensor, 
        offsets=offsets, 
        jagged_dim=1,
        min_seqlen=min_len,
        max_seqlen=max_len,
    )

def coerce_offsets(src, tgt):
    assert torch.eq(src.offsets(), tgt.offsets()).all().item()
    assert src._ragged_idx == tgt._ragged_idx

    def mb_get_size(t):
        return t.shape[0] if t is not None else None

    return torch.nested.nested_tensor_from_jagged(
        src.values(),
        tgt.offsets(),
        None,
        src._ragged_idx,
        mb_get_size(src._min_seqlen_tensor) if tgt._min_seqlen_tensor is None else mb_get_size(tgt._min_seqlen_tensor),
        mb_get_size(src._max_seqlen_tensor) if tgt._max_seqlen_tensor is None else mb_get_size(tgt._max_seqlen_tensor),
    )

def expand_using_offsets(tensor: torch.Tensor, offsets: torch.Tensor):
    '''Expands a 2D tensor, where the second dim is 1, using offsets, and then packs it into a single dimension'''
    assert tensor.shape[0] == (len(offsets) - 1), f"{tensor.shape[0]} != {len(offsets) - 1}"
    assert tensor.dim() == 2, f"Expected 2D tensor, got {tensor.dim()}D tensor"
    assert tensor.shape[1] == 1, f"Expected tensor with second dim 1, got {tensor.shape[1]}"

    # Compute sizes based on offsets
    lengths = offsets.diff()
    max_len = lengths.max().item()
    
    tensor = tensor.expand(-1, max_len).contiguous()

    tensor = torch.nested.narrow(
        tensor, 
        dim=1,
        start=0,
        length=lengths,
        layout=torch.jagged
    ).contiguous()

    tensor, offsets = packed_tensor_from_jagged(tensor)
    return tensor, offsets

def padded_from_jagged(tensor, pad_value=0.0):
    offsets = tensor.offsets()
    padded = torch.nested.to_padded_tensor(tensor, padding=pad_value)
    return padded, offsets

def jagged_from_padded(tensor, offsets, contiguous=True):
    seq_lens = offsets.diff()
    print(seq_lens)
    
    jagged = torch.nested.narrow(tensor, dim=1, start=0, length=seq_lens, layout=torch.jagged)
    if contiguous:
        jagged = jagged.contiguous()

    return jagged