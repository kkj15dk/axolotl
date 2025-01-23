import torch
import torch.nn as nn
import torch.nn.functional as F

# Lets define a helpful benchmarking function:
import torch.utils.benchmark as benchmark
def benchmark_torch_function_in_microseconds(f, *args, **kwargs):
    t0 = benchmark.Timer(
        stmt="f(*args, **kwargs)", globals={"args": args, "kwargs": kwargs, "f": f}
    )
    return t0.blocked_autorange().mean * 1e6

# Lets define the hyper-parameters of our input
batch_size = 32
max_sequence_len = 1024
num_heads = 8
embed_dimension = 64

dtype = torch.float16
device = torch.device("cuda")

query = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)
key = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)
value = torch.rand(batch_size, num_heads, max_sequence_len, embed_dimension, device=device, dtype=dtype)

query_nested = torch.nested.nested_tensor([torch.randn(num_heads, max_sequence_len - i, embed_dimension) for i in range(batch_size)], device=device, layout=torch.jagged, dtype=dtype)
key_nested = torch.nested.nested_tensor([torch.randn(num_heads, max_sequence_len - i, embed_dimension) for i in range(batch_size)], device=device, layout=torch.jagged, dtype=dtype)
value_nested = torch.nested.nested_tensor([torch.randn(num_heads, max_sequence_len - i, embed_dimension) for i in range(batch_size)], device=device, layout=torch.jagged, dtype=dtype)

print("non-nested tensors")
print(query.shape)
print(key.shape)
print(value.shape)

# Lets explore the speed of each of the 3 implementations
from torch.nn.attention import SDPBackend, sdpa_kernel

print(f"The default implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value):.3f} microseconds")

# with sdpa_kernel(SDPBackend.MATH):
#     math_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value)
#     print(f"The math implementation runs in {math_time:.3f} microseconds")

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    try:
        flash_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value)
        print(f"The flash attention implementation runs in {flash_time:.3f} microseconds")
    except RuntimeError:
        print("FlashAttention is not supported. See warnings for reasons.")

with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    try:
        efficient_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query, key, value)
        print(f"The memory efficient implementation runs in {efficient_time:.3f} microseconds")
    except RuntimeError:
        print("EfficientAttention is not supported. See warnings for reasons.")

print("nested tensors")
print(query_nested.shape)
print(key_nested.shape)
print(value_nested.shape)

print(f"The default implementation runs in {benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query_nested, key_nested, value_nested):.3f} microseconds")

# with sdpa_kernel(SDPBackend.MATH):
#     math_time_nested=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query_nested, key_nested, value_nested)
#     print(f"The math implementation runs in {math_time_nested:.3f} microseconds")

with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    try:
        flash_time_nested=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query_nested, key_nested, value_nested)
        print(f"The flash attention implementation runs in {flash_time_nested:.3f} microseconds")
    except RuntimeError:
        print("FlashAttention is not supported. See warnings for reasons.")

with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    try:
        efficient_time=benchmark_torch_function_in_microseconds(F.scaled_dot_product_attention, query_nested, key_nested, value_nested)
        print(f"The memory efficient implementation runs in {efficient_time:.3f} microseconds")
    except RuntimeError:
        print("EfficientAttention is not supported. See warnings for reasons.")