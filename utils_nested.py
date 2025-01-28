import torch

def packed_tensor_from_jagged(tensor):
    assert tensor.is_nested, "Input tensor must be nested"
    return tensor.values(), tensor.offsets()

def jagged_from_packed_tensor(tensor, offsets):
    '''
    Make a jagged tensor from valeus and offsets. Preserves gradient.
    '''
    return torch.nested.nested_tensor_from_jagged(tensor, offsets)

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

def expand_using_offsets(tensor, offsets):
    assert tensor.shape[0] == (len(offsets) - 1), f"{tensor.shape[0]} != {len(offsets) - 1}"

    # Compute sizes based on offsets
    sizes = offsets.diff().tolist()
    shape = tensor.shape
    
    # Expand each segment of the tensor to match the corresponding size
    expanded_segments = [tensor[i].expand(size, *shape[1:]) for i, size in enumerate(sizes)]

    # Concatenate the expanded segments back into a single tensor
    expanded_tensor = torch.cat(expanded_segments, dim=0)
    
    return expanded_tensor

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