import re
from transformers import GPT2TokenizerFast
from itertools import chain
import numpy as np
import torch

import urllib.request
import zipfile
import requests
import json
from datasets import load_from_disk, Dataset

from torch.utils.data import DataLoader, DistributedSampler, Sampler
from typing import Optional, List

def cycle_loader(dataloader):
    while True:
        for data in dataloader:
            yield data
        # go to next epoch
        dataloader.batch_sampler.set_epoch(dataloader.batch_sampler.epoch + 1)


def get_dataset(name):
    dataset = load_from_disk(name)
    dataset = dataset.with_format('torch')

    return dataset


def maybe_truncate(input_ids, max_len, generator=None):
    # Truncate the input_ids if it is longer than max_len
    seq_len = len(input_ids)
    if seq_len > max_len:
        index = torch.randint(0, seq_len - max_len, (1,), generator=generator).item()
        input_ids = input_ids[index:index + max_len]
    return input_ids


def get_dataloaders(train_batch_size,
                    valid_batch_size,
                    ngpus,
                    accum,
                    train_path,
                    valid_path,
                    max_length,
                    drop_last,
                    distributed=True
):
    if train_batch_size % (ngpus * accum) != 0:
            raise ValueError(f"Train Batch Size {train_batch_size} is not divisible by {ngpus} gpus with accumulation {accum}.")
    if valid_batch_size % (ngpus * accum) != 0:
        raise ValueError(f"Eval Batch Size for {valid_batch_size} is not divisible by {ngpus} gpus with accumulation {accum}.")

    train_set = get_dataset(train_path)
    valid_set = get_dataset(valid_path)

    if distributed:
        train_sampler = DistributedSequencePackingSampler(train_set,
                                                          max_length=max_length, # TODO: make sure this gets the right length with distributed, and make it distribute properly
                                                          total_length=max_length * valid_batch_size // (ngpus * accum), # TODO: make sure this gets the right length with distributed, and make it distribute properly
                                                          drop_last=drop_last,
        )
        val_sampler = DistributedSequencePackingSampler(valid_set, 
                                                        max_length=max_length, # TODO: make sure this gets the right length with distributed, and make it distribute properly
                                                        total_length=max_length * valid_batch_size // (ngpus * accum), # TODO: make sure this gets the right length with distributed, and make it distribute properly
                                                        drop_last=drop_last,
        )
    else:
        train_sampler = SequencePackingSampler(train_set,
                                               max_length=max_length, 
                                               total_length=max_length * train_batch_size // (ngpus * accum), 
                                               drop_last=drop_last,
        )
        val_sampler = SequencePackingSampler(valid_set, 
                                             max_length=max_length, 
                                             total_length=max_length * valid_batch_size // (ngpus * accum), 
                                             drop_last=drop_last,
        )

    train_loader = cycle_loader(DataLoader(
        train_set,
        # batch_size=.batch_size // (config.ngpus * .accum),
        batch_sampler=train_sampler,
        num_workers=8,
        collate_fn=train_sampler.collate_fn,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        # batch_size=config.eval.batch_size // (config.ngpus * .accum),
        batch_sampler=val_sampler,
        num_workers=8,
        collate_fn=val_sampler.collate_fn,
        pin_memory=True,
        shuffle=(val_sampler is None),
    ))
    return train_loader, valid_loader


class SequencePackingSampler(Sampler):
    def __init__(self, 
                 dataset,
                 indices: List[int]=None,
                 max_length: int=None, 
                 total_length: int=None,
                 seed=0,
                 drop_last=False,
    ):
        
        self.dataset = dataset # a clustered dataset
        self.max_length = max_length
        self.total_length = total_length
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last

        if indices is None: # Indices should only be used with ditributed sampler
            self.indices = list(range(len(dataset)))
            self.shuffle = True # only shuffle when we are not using distributed sampler. The distributed sampler should handle the shuffling
        else:
            self.indices = indices
            self.shuffle = False # we only need to shuffle when we are not using distributed sampler. The distributed sampler should handle the shuffling

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.indices), generator=g).tolist()
        else:
            indices = self.indices

        batch = []
        batch_length = 0
        for idx in indices:
            cluster_size = self.dataset[idx]['cluster_size'].item()
            if cluster_size == 1:
                cluster_idx = 0
            else:
                # To deterministically sample from the clusters, we use the epoch to index into the cluster, same as in the collate_fn (very important)
                cluster_idx = torch.randint(0, cluster_size, (1,), generator=torch.Generator().manual_seed(self.seed + self.epoch)).item()

            length = self.dataset[idx]['length'][cluster_idx].item()
            length = min(length, self.max_length)
            
            batch.append(idx)
            batch_length += length

            if batch_length >= self.total_length:
                yield batch
                batch = []
                batch_length = 0

        if len(batch) > 0 and not self.drop_last:
            yield batch


    def collate_fn(self, batch):
        g = torch.Generator().manual_seed(self.seed + self.epoch)

        # get the sizes of the clusters
        cluster_sizes = [x['cluster_size'] for x in batch]
        
        indexes = []
        for cs in cluster_sizes:
            if cs == 1:
                indexes.append(0)
                continue
            idx = torch.randint(0, cs, (1,), generator=torch.Generator().manual_seed(self.seed + self.epoch)).item()
            indexes.append(idx)

        # get the input_ids for each cluster
        input_ids_list = [x["input_ids"][i] for x, i in zip(batch, indexes)]
        input_ids_list = [maybe_truncate(input_ids, self.max_length, generator=g) for input_ids in input_ids_list[:-1]]

        # the last input ids should be truncated to the remaining length, to not exceed the total length
        length_sum = sum([min(x["length"][i].item(), self.max_length) for x, i in zip(batch, indexes[:-1])])
        last_length = min(self.total_length - length_sum, self.max_length)
        # print("total length: ", length_sum + last_length)

        input_ids_list.append(maybe_truncate(input_ids_list[-1], last_length, generator=g))

        # convert to nested tensor for the model
        input_ids = torch.nested.nested_tensor(input_ids_list, layout=torch.jagged)
        label = torch.tensor([x["label"][i] for x, i in zip(batch, indexes)])

        return {"input_ids": input_ids, "label": label}

    def __len__(self):
        raise NotImplementedError("SequencePackingSampler does not support __len__")
    
    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch


class DistributedSequencePackingSampler(DistributedSampler):
    def __init__(self, 
                 dataset, 
                 num_replicas=None, 
                 rank=None, 
                 shuffle=True, 
                 max_length: int=None, # max length of each sequence in the batch
                 total_length: int=None, # total length of the batch (total amount of tokens)
                 seed=0,
                 drop_last=False,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=False)

        self.max_length = max_length
        self.total_length = total_length
        self.seed = seed
        self.dataset = dataset
        self.drop_last = drop_last
        self.epoch = 0

    def __iter__(self):
        self.indices = list(super().__iter__())
        batch_sampler = SequencePackingSampler(self.dataset, 
                                               indices=self.indices, 
                                               max_length=self.max_length, 
                                               total_length=self.total_length, 
                                               seed=self.seed,
                                               drop_last=self.drop_last,
        )
        batch_sampler.set_epoch(self.epoch) # set the epoch to the epoch of the distributed sampler
        self.collate_fn = batch_sampler.collate_fn # set the collate_fn to the collate_fn of the SequencePackingSampler
        return iter(batch_sampler)

    def collate_fn(self, batch):
        raise NotImplementedError("DistributedSequencePackingSampler does not support collate_fn. It should be updated in __iter__ by the SequencePackingSampler")

    def __len__(self):
        raise NotImplementedError("DistributedSequencePackingSampler does not support __len__")

    def set_epoch(self, epoch: int) -> None:
        r"""
        Set the epoch for this sampler.

        When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch