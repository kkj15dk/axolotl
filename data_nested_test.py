# %%
import os
import numpy as np
import torch.distributed as dist
import datetime
import torch

import axolotl.data_nested as data

# Build data iterators
train_ds, eval_ds = data.get_dataloaders(4,
                                         4,
                                         1,
                                         1,
                                         '/home/kkj/axolotl/datasets/IPR036736_90_grouped/train',
                                         '/home/kkj/axolotl/datasets/IPR036736_90_grouped/valid',
                                         512,
                                         True,
                                         distributed=False,
)
train_iter = iter(train_ds)
eval_iter = iter(eval_ds)

# %%
i = 0
while True:
    data = next(train_iter)
    i += 1
    if i % 1000 == 0:
        print(i, data)