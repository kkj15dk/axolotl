import torch
import time

# Define the modulate function
def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if x.is_nested:
        # Convert shift and scale to nested tensors
        shift_nested = torch.nested.nested_tensor([sh.expand_as(t) for t, sh in zip(x.unbind(), shift.unbind())])
        scale_nested = torch.nested.nested_tensor([sc.expand_as(t) for t, sc in zip(x.unbind(), scale.unbind())])
        
        # Apply modulation using broadcasting
        return x * (1 + scale_nested) + shift_nested
    else:
        return x * (1 + scale) + shift

def modulate_fused(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return modulate(x, shift, scale)

# Create sample data
batch_size = 32
seq_len = 128
dim = 512

x = torch.randn(batch_size, seq_len, dim, device='cuda')
shift = torch.randn(batch_size, 1, dim, device='cuda')
scale = torch.randn(batch_size, 1, dim, device='cuda')

# Benchmark regular tensors
start_time = time.time()
for _ in range(1000):
    modulate_fused(x, shift, scale)
print("Regular tensors time:", time.time() - start_time)

# Create nested tensors
x_nested = torch.nested.nested_tensor([torch.randn(seq_len, dim, device='cuda') for _ in range(batch_size)])
shift_nested = torch.nested.nested_tensor([torch.randn(1, dim, device='cuda') for _ in range(batch_size)])
scale_nested = torch.nested.nested_tensor([torch.randn(1, dim, device='cuda') for _ in range(batch_size)])

# Benchmark nested tensors
start_time = time.time()
for _ in range(1000):
    modulate_fused(x_nested, shift_nested, scale_nested)
print("Nested tensors time:", time.time() - start_time)