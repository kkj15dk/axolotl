import torch
import torch.nn as nn

def packed_tensor_from_jagged(tensor):
    offsets = tensor.offsets()
    return torch.cat([t for t in tensor.unbind()], dim = 0), offsets

def modulate(x, shift, scale):

    print("x: ", x.shape)
    if scale is not None:
        print("scale: ", scale.shape)
        x = x * (1 + scale)
    if shift is not None:
        print("shift: ", shift.shape)
        x = x + shift
    return x

class test_model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.modulation = nn.Linear(dim, 2 * dim)

    def forward(self, x, c): # x is a ragged tensor (batch_size=4, j, dim=64), c is a regular tensor (batch_size=4, dim=64)
        shift, scale = self.modulation(c).chunk(2, dim=-1)
        shift, scale = shift.unsqueeze(1), scale.unsqueeze(1) # I think it has something to do with this unsqueeze

        return modulate(x, shift, scale)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 128
max_len = 4096
dim = 64 * 24

model = test_model(dim).to(device)

# ### Normal tensors
# batch =torch.randn(batch_size, max_len, dim, device=device) # batch_size=4, j=512, dim=64
# c = torch.randn(batch_size, dim, device=device) # batch_size=4, dim=64

# print("Normal tensors")
# output = model(batch, c)
# loss = output.sum(dim=-1).mean()
# loss.backward()
###

### Nested tensors
batch = torch.nested.nested_tensor([torch.randn(max_len - i, dim) for i in range(batch_size)], device=device, layout=torch.jagged) # batch_size=4, j=jagged, dim=64
c = torch.randn(batch_size, dim, device=device) # batch_size=4, dim=64

print("Nested tensors")
for i in range(10):
    print(i)
    output = model(batch, c)
    output, offsets = packed_tensor_from_jagged(output)
    loss = output.sum(dim=-1).mean()
    loss.backward() 
###