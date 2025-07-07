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

import numpy as np
import numba

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
                                                          epoch=epoch,
                                                          drop_last=drop_last,
        )
        val_sampler = SimpleDistributedBatchSampler(valid_set, 
                                                        max_length=max_length, # TODO: make sure this gets the right length with distributed, and make it distribute properly
                                                        total_length=max_length * valid_batch_size // (ngpus * accum), # TODO: make sure this gets the right length with distributed, and make it distribute properly
                                                        epoch=epoch,
                                                        drop_last=drop_last,
        )
    else:
        train_sampler = SequencePackingSampler(train_set,
                                               max_length=max_length, 
                                               total_length=max_length * train_batch_size // (ngpus * accum), 
                                               epoch=epoch,
                                               drop_last=drop_last,
        )
        val_sampler = SequencePackingSampler(valid_set, 
                                             max_length=max_length, 
                                             total_length=max_length * valid_batch_size // (ngpus * accum), 
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


class SequencePackingSampler(Sampler):
    def __init__(self, 
                 dataset,
                 indices: List[int]=None,
                 max_length: int=None, 
                 total_length: int=None,
                 seed=0,
                 drop_last=False,
                 epoch=0,
    ):
        
        self.dataset = dataset # a clustered dataset
        self.max_length = max_length
        self.total_length = total_length
        self.seed = seed
        self.epoch = epoch
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
            indices = self.indices # leave the shuffling to the distributed sampler

        batch = []
        batch_length = 0
        i_pseudo_random = 0 # to deterministically sample from the clusters, we use a pseudo random number generator with a seed that is incremented for each sample in the batch
        for idx in indices:
            cluster_size = self.dataset[idx]['cluster_size'].item()
            if cluster_size == 1:
                cluster_idx = 0
            else:
                # To deterministically sample from the clusters, we use a pseudo random number generator with a seed that is incremented for each sample in the batch, this is same as in the collate_fn (very important)
                cluster_idx = torch.randint(0, cluster_size, (1,), generator=torch.Generator().manual_seed(self.seed + i_pseudo_random)).item()

            length = self.dataset[idx]['length'][cluster_idx].item()
            length = min(length, self.max_length)
            
            batch.append(idx)
            batch_length += length
            i_pseudo_random += 1

            if batch_length >= self.total_length:
                yield batch
                # reset the batch
                batch = []
                batch_length = 0
                i_pseudo_random = 0

        if len(batch) > 0 and not self.drop_last:
            # print("last batch of total length: ", batch_length)
            yield batch


    def collate_fn(self, batch):
        g = torch.Generator().manual_seed(self.seed)
        
        indexes = []
        for i_pseudo_random, x in enumerate(batch):
            cluster_size = x['cluster_size'].item()
            if cluster_size == 1:
                indexes.append(0)
                continue
            # get the index of the cluster to sample from
            idx = torch.randint(0, cluster_size, (1,), generator=torch.Generator().manual_seed(self.seed + i_pseudo_random)).item()
            indexes.append(idx)
        
        # get the input_ids for each cluster
        input_ids_list = [x["input_ids"][i] for x, i in zip(batch, indexes)]
        input_ids_list = [maybe_truncate(input_ids, self.max_length, generator=g) for input_ids in input_ids_list[:-1]]

        # the last input ids should be truncated to the remaining length, to not exceed the total length
        length_sum = sum([min(x["length"][i].item(), self.max_length) for x, i in zip(batch, indexes[:-1])])
        last_length = min(self.total_length - length_sum, self.max_length)

        input_ids_list.append(maybe_truncate(input_ids_list[-1], last_length, generator=g))

        # convert to nested tensor for the model
        input_ids = torch.nested.nested_tensor(input_ids_list, layout=torch.jagged)
        label = torch.tensor([x["label"][i] for x, i in zip(batch, indexes)])

        return {"input_ids": input_ids, "label": label, "batch": batch}

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
                 epoch=0,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=False)

        self.max_length = max_length
        self.total_length = total_length
        self.seed = seed
        self.dataset = dataset
        self.drop_last = drop_last
        self.epoch = epoch

    def __iter__(self):
        self.indices = list(super().__iter__()) # use the indices from the distributed sampler
        batch_sampler = SequencePackingSampler(self.dataset, 
                                               indices=self.indices, 
                                               max_length=self.max_length, 
                                               total_length=self.total_length, 
                                               seed=self.seed,
                                               drop_last=self.drop_last,
                                               epoch=self.epoch, # set the epoch to the epoch of the distributed sampler
        )
        self.collate_fn = batch_sampler.collate_fn # set the collate_fn to the collate_fn of the SequencePackingSampler
        return iter(batch_sampler)

    def collate_fn(self, batch):
        raise NotImplementedError("DistributedSequencePackingSampler does not support collate_fn. It should be updated in __iter__ by the SequencePackingSampler")

    def __len__(self):
        raise NotImplementedError("DistributedSequencePackingSampler does not support __len__")
    

### Multipack sampler
### https://github.com/imoneoi/multipack_sampler/blob/master/multipack_sampler.py


@numba.njit
def lpt_check(heap: np.ndarray, A: np.ndarray, c: int, n: int):
    # LPT (Longest processing time first scheduling)
    # Time: O(|A| log |A| + |A| log n)

    A = np.sort(A)[::-1]
    heap.fill(0)
    for size in A:
        # Put into smallest element
        heap[1] += size
        if heap[1] > c:
            return False

        # Heapify (Sink)
        # https://stackoverflow.com/questions/20397674/replacing-element-in-min-heap
        u = 1
        while (u << 1) <= n:
            v = u << 1  # lch
            rch = (u << 1) | 1
            if rch <= n and heap[rch] < heap[v]:
                v = rch
            
            if heap[u] <= heap[v]:
                break

            heap[u], heap[v] = heap[v], heap[u]
            u = v

    return True


@numba.njit
def lpt_with_result(heap: np.ndarray, A: np.ndarray, n: int, start_index: int, rank: int):
    # LPT (Longest processing time first scheduling)
    # Time: O(|A| log |A| + |A| log n)

    result = []

    indices = np.argsort(A)[::-1]
    A = A[indices]

    heap.fill(0)
    heap_id = np.arange(-1, n, dtype=A.dtype)
    for idx, size in enumerate(A):
        # Put into smallest element
        heap[1] += size
        if heap_id[1] == rank:
            result.append(start_index + indices[idx])

        # Heapify (Sink)
        # https://stackoverflow.com/questions/20397674/replacing-element-in-min-heap
        u = 1
        while (u << 1) <= n:
            v = u << 1  # lch
            rch = (u << 1) | 1
            if rch <= n and heap[rch] < heap[v]:
                v = rch
            
            if heap[u] <= heap[v]:
                break

            heap[u], heap[v] = heap[v], heap[u]
            heap_id[u], heap_id[v] = heap_id[v], heap_id[u]
            u = v

    return result


@numba.njit
def allocate_multipack(lengths: np.ndarray, lengths_cumsum: np.ndarray, rank: int, c: int, n: int):
    # Dynamic batch allocator, binary search + LPT
    # ~99.5% efficiency on OpenChat training set (12 * 2048 ctx len)

    heap = np.zeros(n + 1, dtype=lengths.dtype)

    s = 0
    start_index = 0
    result = []

    while True:
        # binary search [l, r)
        l = 1
        r = 1 + np.searchsorted(lengths_cumsum[start_index:], s + c * n, "right")

        while r - l > 1:
            m = (l + r) // 2
            if lpt_check(heap, lengths[start_index: start_index + m], c, n):
                l = m
            else:
                r = m

        # use length l
        if l < n:
            break  # Can't allocate each sequence to a single machine

        batch = lpt_with_result(heap, lengths[start_index: start_index + l], n, start_index, rank)

        start_index += l
        s = lengths_cumsum[start_index - 1]

        # add local rank
        result.append(batch)

    return result, s, len(result) * c * n


class MultipackDistributedBatchSampler(Sampler):
    """Unpadded length sampling using Multipack V2, for models with quadratic attention complexity.
       It also tries to evenly distribute the sequences using LPT, so that quadratic load is more balanced.

       Approximate (at most 1.33x ?) the optimal solution of the identical-machines scheduling problem, which is NP-hard.

       Time Complexity: O(n log n log k)
       n = maximum number of sequences per batch, k = number of nodes
    """
    def __init__(
        self,
        lengths: List[torch.Tensor], # a list of tensors, containing the lengths of sequences in each cluster
        cluster_sizes: torch.Tensor, # a tensor of cluster sizes, where each cluster size corresponds to the number of sequences in the cluster
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        max_length: int = None, # maximum length of each sequence in the batch, what the sequences will be truncated to
        total_length: int = None, # total length of the batch (total amount of tokens)
        seed: int = 0,
        epoch: int = 0,
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

        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.max_length = max_length # maximum length of each sequence in the batch, what the sequecnes will be truncated to

        self.total_length = total_length
        self.lengths = lengths # a list of tensors, where each tensor contains the lengths of the sequences in the cluster
        assert isinstance(self.lengths, list) and all(isinstance(length_tensor, torch.Tensor) for length_tensor in self.lengths), "Lengths should be a list of tensors"
        if cluster_sizes is None:
            # If no cluster sizes are provided, assume each cluster has the same size as the number of sequences in the cluster
            self.cluster_sizes = [len(length_list) for length_list in self.lengths]
        else:
            assert isinstance(cluster_sizes, torch.Tensor), "Cluster sizes should be a tensor of integers"
            self.cluster_sizes = cluster_sizes

        self.epoch = epoch

        # statistics
        self.eff_total_used = 0
        self.eff_total_slots = 0

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def generate_batches(self, set_stats=False):
        indices = np.random.Generator(np.random.Philox(seed=self.seed + self.epoch)).permutation(len(self.lengths))

        # Get the lengths of the sequences in the batch, truncated to the maximum length. 
        # We use the seed and epoch to deterministically select one sequence from each cluster
        cluster_selection_seed = (self.seed + self.epoch)
        
        # Optimized version using numpy vectorization TODO: optimize this further
        cluster_indices = np.array([cluster_selection_seed % size for size in self.cluster_sizes], dtype=np.int32)
        selected_lengths = np.array([length_list[idx] for length_list, idx in zip(self.lengths, cluster_indices)], dtype=np.int32)
        lengths = np.minimum(selected_lengths, self.max_length)
        
        lengths = lengths[indices]

        lengths_cumsum = np.cumsum(lengths)

        batches, total_used, total_slots = allocate_multipack(lengths=lengths,
                                                              lengths_cumsum=lengths_cumsum,
                                                              rank=self.rank,
                                                              c=self.total_length,
                                                              n=self.num_replicas)
        
        batches = [indices[batch] for batch in batches]

        # statistics
        if set_stats:
            self.eff_total_used += total_used
            self.eff_total_slots += total_slots
        
        return batches
    
    def __iter__(self):
        batches = self.generate_batches(set_stats=True)
        return iter(batches)

    def num_batches(self):
        batches = self.generate_batches()
        return len(batches)

    def efficiency(self):
        return self.eff_total_used / self.eff_total_slots
    
    def collate_fn(self, batch):

        g = torch.Generator().manual_seed(self.seed + self.epoch)
        cluster_selection_seed = (self.seed + self.epoch)

        input_ids_list = []
        label_list = []

        for x in batch:
            cluster_size = x['cluster_size'].item()
            if cluster_size == 1:
                idx = 0
                continue
            idx = (cluster_selection_seed % cluster_size)

            input_ids_list.append(maybe_truncate(x["input_ids"][idx], self.max_length, generator=g))
            label_list.append(x["label"][idx])

        # convert to nested tensor for the model
        input_ids = torch.nested.nested_tensor(input_ids_list, layout=torch.jagged)
        label = torch.tensor(label_list, dtype=torch.int64)

        return {"input_ids": input_ids, "label": label, "batch": batch}
    

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
            
            # Add sample to current batch
            batch.append(idx)
            batch_length += truncated_length
            
            # Check if adding this sample caused overflow
            if batch_length >= self.total_length:
                # Check if this batch belongs to our rank
                if batch_count % self.num_replicas == self.rank:
                    yield batch
                
                # Reset for next batch
                batch = []
                batch_length = 0
                batch_count += 1
        
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

        for i in range(len(batch) - 1):
            # Get the sample from the dataset
            cluster_size = batch[i][self.cluster_size_key].item()
            if cluster_size == 1:
                idx = 0
            else:
                idx = (cluster_selection_seed % cluster_size)

            input_ids_list.append(maybe_truncate(batch[i]["input_ids"][idx], self.max_length, generator=g))
            length_sum += min(batch[i][self.length_key][idx].item(), self.max_length)
            label_list.append(batch[i]["label"][idx])

        # Process the last element - calculate its cluster index properly
        last_cluster_size = batch[-1][self.cluster_size_key].item()
        if last_cluster_size == 1:
            last_idx = 0
        else:
            last_idx = (cluster_selection_seed % last_cluster_size)

        # the last input ids should be truncated to the remaining length, to not exceed the total length
        last_length = min(self.total_length - length_sum, self.max_length)
        input_ids_list.append(maybe_truncate(batch[-1]["input_ids"][last_idx], last_length, generator=g))
        label_list.append(batch[-1]["label"][last_idx])

        # convert to nested tensor for the model
        input_ids = torch.nested.nested_tensor(input_ids_list, layout=torch.jagged)
        label = torch.tensor(label_list, dtype=torch.long)

        return {"input_ids": input_ids, "label": label}

class SimpleBatchSampler(Sampler):

    def __init__(
        self,
        dataset, # A clustered dataset. TODO: specify format
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
            
            # Add sample to current batch
            batch.append(idx)
            batch_length += truncated_length
            
            # Check if adding this sample caused overflow
            if batch_length >= self.total_length:
                yield batch
                
                # Reset for next batch
                batch = []
                batch_length = 0
                batch_count += 1
        
        # Handle remaining batch if not empty
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def collate_fn(self, batch): # TODO: why are there a lot of batches with less than max amount of tokens?
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        cluster_selection_seed = (self.seed + self.epoch)

        input_ids_list = []
        label_list = []
        length_sum = 0

        for i in range(len(batch) - 1):
            # Get the sample from the dataset
            cluster_size = batch[i][self.cluster_size_key].item()
            if cluster_size == 1:
                idx = 0
            else:
                idx = (cluster_selection_seed % cluster_size)

            input_ids_list.append(maybe_truncate(batch[i]["input_ids"][idx], self.max_length, generator=g))
            length_sum += min(batch[i][self.length_key][idx].item(), self.max_length)
            label_list.append(batch[i]["label"][idx])

        # Process the last element - calculate its cluster index properly
        last_cluster_size = batch[-1][self.cluster_size_key].item()
        if last_cluster_size == 1:
            last_idx = 0
        else:
            last_idx = (cluster_selection_seed % last_cluster_size)

        # the last input ids should be truncated to the remaining length, to not exceed the total length
        last_length = min(self.total_length - length_sum, self.max_length)
        input_ids_list.append(maybe_truncate(batch[-1]["input_ids"][last_idx], last_length, generator=g))
        label_list.append(batch[-1]["label"][last_idx])

        # convert to nested tensor for the model
        input_ids = torch.nested.nested_tensor(input_ids_list, layout=torch.jagged)
        label = torch.tensor(label_list, dtype=torch.long)

        return {"input_ids": input_ids, "label": label}