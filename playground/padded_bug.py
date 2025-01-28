import torch
import torch.nn as nn

def padded_from_jagged(tensor, pad_value=0.0):
    offsets = tensor.offsets()
    padded = torch.nested.to_padded_tensor(tensor, padding=pad_value)
    return padded, offsets

def jagged_from_padded(tensor, offsets, contiguous=True):
    seq_lens = offsets.diff()
    
    jagged = torch.nested.narrow(tensor, dim=1, start=0, length=seq_lens, layout=torch.jagged)
    if contiguous:
        jagged = jagged.contiguous()

    return jagged

class test_model(nn.Module):
    def __init__(self, dim, max_len):
        super().__init__()
        self.pos_emb = nn.Parameter(torch.randn(1, max_len, dim), requires_grad=True)

    def forward(self, x): # x is a ragged tensor (batch_size=4, j, dim=64), c is a regular tensor (batch_size=4, dim=64)
            
        for i in range(10):
            x_padded, offsets = padded_from_jagged(x)
            x_padded = x_padded * self.pos_emb
            x = jagged_from_padded(x_padded, offsets, contiguous=True)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = 4
max_len = 4096
dim = 64

model = test_model(dim, max_len).to(device)

batch = torch.nested.nested_tensor([torch.randn(max_len - i, dim) for i in range(batch_size)], device=device, layout=torch.jagged) # batch_size=4, j=jagged, dim=64

output = model(batch)
loss = output.mean()
loss.backward()