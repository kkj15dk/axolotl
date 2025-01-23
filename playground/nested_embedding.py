import torch
import torch.nn as nn

def packed_tensor_from_jagged(tensor):
    offsets = tensor.offsets()
    return torch.cat([t for t in tensor.unbind()], dim = 0), offsets

class test_model(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.embedding = nn.Embedding(20, dim)

    def forward(self, x):
        print(x.shape)
        x = self.embedding(x)
        print(x.shape)

        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = test_model(64).to(device)

### Bug here, when using nested tensors
batch_list = [torch.randint(0, 20, (l,), dtype=torch.long) for l in [64, 128, 256, 512]]
batch = torch.nested.nested_tensor(batch_list, layout=torch.jagged, device=device)

output = model(batch)
output, offsets = packed_tensor_from_jagged(output)
loss = output.sum(dim=-1).mean()
loss.backward()