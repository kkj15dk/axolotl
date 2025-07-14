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
from typing import Optional, List, Union

import torch.distributed as dist
from torch.utils.data import Sampler

class CyclingDataLoader:
    """A wrapper that cycles through a DataLoader infinitely while managing epochs"""
    
    def __init__(self, dataloader: DataLoader, shuffle_each_epoch: bool = True):
        self.dataloader = dataloader
        self.shuffle_each_epoch = shuffle_each_epoch
        
    @property
    def batch_sampler(self):
        """Access to the underlying batch sampler"""
        return self.dataloader.batch_sampler
    
    @property
    def epoch(self):
        """Current epoch of the batch sampler"""
        return self.dataloader.batch_sampler.epoch
    
    def __iter__(self):
        return self._cycle_generator()
    
    def _cycle_generator(self):
        while True:
            # iterate over the dataloader
            # print("Starting dataloader...")
            # print("first batch: ", 0, next(iter(dataloader))['input_ids'].offsets())
            for i, data in enumerate(self.dataloader):
                yield data
            # go to next epoch
            if self.shuffle_each_epoch:
                self.dataloader.batch_sampler.set_epoch(self.dataloader.batch_sampler.epoch + 1)
            print("End of dataloader, restarting...")
            print("We are now at epoch: ", self.dataloader.batch_sampler.epoch)
            # print("last batch: ", i, data['input_ids'].offsets())

def cycle_loader(dataloader: DataLoader, shuffle_each_epoch: bool = True):
    """Create a cycling dataloader that can be iterated infinitely"""
    return CyclingDataLoader(dataloader, shuffle_each_epoch)


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
                    num_workers,
                    distributed=True,
                    seed=42, # Seed for picking the data in the clusters and making torch generators.
                    epoch=0,
                    shuffle_each_epoch=True,
):
    if train_batch_size % (ngpus * accum) != 0:
            raise ValueError(f"Train Batch Size {train_batch_size} is not divisible by {ngpus} gpus with accumulation {accum}.")
    if valid_batch_size % (ngpus * accum) != 0:
        raise ValueError(f"Eval Batch Size for {valid_batch_size} is not divisible by {ngpus} gpus with accumulation {accum}.")

    train_set = get_dataset(train_path)
    valid_set = get_dataset(valid_path)

    if distributed:
        # train_sampler = DistributedSequencePackingSampler(train_set,
        #                                                   max_length=max_length, # TODO: make sure this gets the right length with distributed, and make it distribute properly
        #                                                   total_length=max_length * train_batch_size // (ngpus * accum), # TODO: make sure this gets the right length with distributed, and make it distribute properly
        #                                                   epoch=epoch,
        #                                                   drop_last=drop_last,
        # )
        # val_sampler = DistributedSequencePackingSampler(valid_set, 
        #                                                 max_length=max_length, # TODO: make sure this gets the right length with distributed, and make it distribute properly
        #                                                 total_length=max_length * valid_batch_size // (ngpus * accum), # TODO: make sure this gets the right length with distributed, and make it distribute properly
        #                                                 epoch=epoch,
        #                                                 drop_last=drop_last,
        # )
        train_sampler = SimpleDistributedBatchSampler(train_set,
                                                          max_length=max_length, # TODO: make sure this gets the right length with distributed, and make it distribute properly
                                                          total_length=max_length * train_batch_size // (ngpus * accum), # TODO: make sure this gets the right length with distributed, and make it distribute properly
                                                          seed=seed,
                                                          epoch=epoch,
                                                          drop_last=drop_last,
        )
        val_sampler = SimpleDistributedBatchSampler(valid_set, 
                                                        max_length=max_length, # TODO: make sure this gets the right length with distributed, and make it distribute properly
                                                        total_length=max_length * valid_batch_size // (ngpus * accum), # TODO: make sure this gets the right length with distributed, and make it distribute properly
                                                        seed=seed,
                                                        epoch=epoch,
                                                        drop_last=drop_last,
        )
    else:
        train_sampler = SimpleBatchSampler(train_set,
                                               max_length=max_length, 
                                               total_length=max_length * train_batch_size // (ngpus * accum), 
                                               seed=seed,
                                               epoch=epoch,
                                               drop_last=drop_last,
        )
        val_sampler = SimpleBatchSampler(valid_set, 
                                             max_length=max_length, 
                                             total_length=max_length * valid_batch_size // (ngpus * accum), 
                                             seed=seed,
                                             epoch=epoch,
                                             drop_last=drop_last,
        )

    train_loader = cycle_loader(DataLoader(
        train_set,
        # batch_size=.batch_size // (config.ngpus * .accum),
        batch_sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=train_sampler.collate_fn,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=(num_workers > 0),
    ),
    shuffle_each_epoch=shuffle_each_epoch,  # No need to shuffle validation data
    )
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        # batch_size=config.eval.batch_size // (config.ngpus * .accum),
        batch_sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=val_sampler.collate_fn,
        pin_memory=True,
        shuffle=(val_sampler is None),
        persistent_workers=(num_workers > 0),
    ),
    shuffle_each_epoch=shuffle_each_epoch,  # No need to shuffle validation data
    )
    return train_loader, valid_loader


## This is the one ## I think...

class SimpleDistributedBatchSampler(Sampler):

    def __init__(
        self,
        dataset, # A clustered dataset TODO specify format
        max_length: int, # maximum length of each sequence in the batch, what the sequences will be truncated to
        total_length: int, # total length of the batch (total amount of tokens)
        length_key = 'length', # a list of tensors, containing the lengths of sequences in each cluster
        cluster_size_key = 'cluster_size', # a tensor of cluster sizes, where each cluster size corresponds to the number of sequences in the cluster
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        seed: int = 0,
        epoch: int = 0,
        shuffle: bool = True,
        drop_last: bool = False, # whether to drop the last batch if it is smaller than the total length
    ):
        # Get rank
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.dataset = dataset
        self.length_key = length_key
        self.cluster_size_key = cluster_size_key
        self.max_length = max_length # maximum length of each sequence in the batch, what the sequecnes will be truncated to
        self.total_length = total_length

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.epoch = epoch
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.indices = list(range(len(dataset)))

    def set_epoch(self, epoch: int):
        self.epoch = epoch
    
    def __iter__(self):
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.indices), generator=g).tolist()
        else:
            indices = self.indices.copy()

        # We use the seed and epoch to deterministically select one sequence from each cluster
        cluster_selection_seed = (self.seed + self.epoch)
        
        # Track batch generation for distributed processing
        batch = []
        batch_length = 0
        batch_count = 0  # Global batch counter
        
        for idx in indices:
            # Get cluster size and select sequence deterministically
            cluster_size = self.dataset[idx][self.cluster_size_key].item()
            cluster_idx = cluster_selection_seed % cluster_size
            
            # Get the length and truncate to max_length
            sequence_length = self.dataset[idx][self.length_key][cluster_idx].item()
            truncated_length = min(sequence_length, self.max_length)
            
            # Increment length for the batch
            batch_length += truncated_length
            
            # Check if adding this sample would cause overflow
            if batch_length > self.total_length:
                # Check if this batch belongs to our rank
                if batch_count % self.num_replicas == self.rank:
                    yield batch
                
                # Reset for next batch
                batch = []
                batch_length = truncated_length
                batch_count += 1

            # Add sample to the batch
            batch.append(idx)
        
        # Handle remaining batch if not empty
        if len(batch) > 0 and not self.drop_last:
            # Check if this final batch belongs to our rank
            if batch_count % self.num_replicas == self.rank:
                yield batch

    def collate_fn(self, batch): # TODO: why are there a lot of batches with less than max amount of tokens?
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        cluster_selection_seed = (self.seed + self.epoch)

        input_ids_list = []
        label_list = []
        length_sum = 0

        for x in batch:
            # Get the sample from the dataset
            cluster_size = x[self.cluster_size_key].item()
            if cluster_size == 1:
                idx = 0
            else:
                idx = (cluster_selection_seed % cluster_size)

            input_ids_list.append(maybe_truncate(x["input_ids"][idx], self.max_length, generator=g))
            length_sum += min(x[self.length_key][idx].item(), self.max_length)
            label_list.append(x["label"][idx])

        # convert to nested tensor for the model
        input_ids = torch.nested.nested_tensor(input_ids_list, layout=torch.jagged)
        label = torch.tensor(label_list, dtype=torch.long)

        return {"input_ids": input_ids, "label": label}


class SimpleBatchSampler(Sampler):

    def __init__(
        self,
        dataset, # A clustered dataset TODO specify format
        max_length: int, # maximum length of each sequence in the batch, what the sequences will be truncated to
        total_length: int, # total length of the batch (total amount of tokens)
        length_key = 'length', # a list of tensors, containing the lengths of sequences in each cluster
        cluster_size_key = 'cluster_size', # a tensor of cluster sizes, where each cluster size corresponds to the number of sequences in the cluster
        seed: int = 0,
        epoch: int = 0,
        shuffle: bool = True,
        drop_last: bool = False, # whether to drop the last batch if it is smaller than the total length
    ):
        self.dataset = dataset
        self.length_key = length_key
        self.cluster_size_key = cluster_size_key
        self.max_length = max_length # maximum length of each sequence in the batch, what the sequecnes will be truncated to
        self.total_length = total_length

        self.seed = seed
        self.epoch = epoch
        self.shuffle = shuffle
        self.drop_last = drop_last

        self.indices = list(range(len(dataset)))

    def set_epoch(self, epoch: int):
        self.epoch = epoch
    
    def __iter__(self):
        if self.shuffle:
            g = torch.Generator().manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.indices), generator=g).tolist()
        else:
            indices = self.indices.copy()

        # We use the seed and epoch to deterministically select one sequence from each cluster
        cluster_selection_seed = (self.seed + self.epoch)
        
        # Track batch generation for distributed processing
        batch = []
        batch_length = 0
        batch_count = 0  # Global batch counter
        
        for idx in indices:
            # Get cluster size and select sequence deterministically
            cluster_size = self.dataset[idx][self.cluster_size_key].item()
            cluster_idx = cluster_selection_seed % cluster_size
            
            # Get the length and truncate to max_length
            sequence_length = self.dataset[idx][self.length_key][cluster_idx].item()
            truncated_length = min(sequence_length, self.max_length)
            
            # Increment length for the batch
            batch_length += truncated_length
            
            # Check if adding this sample would cause overflow
            if batch_length > self.total_length:
                yield batch
                
                # Reset for next batch
                batch = []
                batch_length = truncated_length
                batch_count += 1

            # Add sample to the batch
            batch.append(idx)
        
        # Handle remaining batch if not empty
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def collate_fn(self, batch): # TODO: why are there a lot of batches with less than max amount of tokens?
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        cluster_selection_seed = (self.seed + self.epoch)

        input_ids_list = []
        label_list = []
        length_sum = 0

        for x in batch:
            # Get the sample from the dataset
            cluster_size = x[self.cluster_size_key].item()
            if cluster_size == 1:
                idx = 0
            else:
                idx = (cluster_selection_seed % cluster_size)

            input_ids_list.append(maybe_truncate(x["input_ids"][idx], self.max_length, generator=g))
            length_sum += min(x[self.length_key][idx].item(), self.max_length)
            label_list.append(x["label"][idx])

        # convert to nested tensor for the model
        input_ids = torch.nested.nested_tensor(input_ids_list, layout=torch.jagged)
        label = torch.tensor(label_list, dtype=torch.long)

        return {"input_ids": input_ids, "label": label}