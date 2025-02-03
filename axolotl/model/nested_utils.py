import torch
import timeit


def flatten_nested_tensor(nested_tensor: torch.Tensor, verbose=False):
    # Flatten the nested tensor by concatenating the sub-tensors
    if verbose:
        print("Nested Tensor:")
        print(nested_tensor.shape)
        time_s = timeit.default_timer()
    flat_tensor = torch.cat([t.flatten(0, 0) for t in nested_tensor.unbind()], dim=0)
    if verbose:
        print("Flat Tensor:")
        print(flat_tensor.shape)
        time_e = timeit.default_timer()
        print("Time to flatten: ", time_e - time_s)
    return flat_tensor

def expand_using_offsets(tensor: torch.Tensor, offsets):
    # Compute sizes based on offsets
    sizes = offsets.diff().tolist()
    shape = tensor.shape
    
    # Expand each segment of the tensor to match the corresponding size
    expanded_segments = [tensor[i].expand(size, *shape[1:]) for i, size in enumerate(sizes)]

    # Concatenate the expanded segments back into a single tensor
    expanded_tensor = torch.cat(expanded_segments, dim=0)
    
    return expanded_tensor