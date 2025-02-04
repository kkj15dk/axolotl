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
from typing import Optional

def cycle_loader(dataloader):
    while True:
        for data in dataloader:
            yield data
        # go to next epoch
        dataloader.batch_sampler.set_epoch(dataloader.batch_sampler.epoch + 1)


def wt_detokenizer(string):
    # contractions
    string = string.replace("s '", "s'")
    string = re.sub(r"/' [0-9]/", r"/'[0-9]/", string)
    # number separators
    string = string.replace(" @-@ ", "-")
    string = string.replace(" @,@ ", ",")
    string = string.replace(" @.@ ", ".")
    # punctuation
    string = string.replace(" : ", ": ")
    string = string.replace(" ; ", "; ")
    string = string.replace(" . ", ". ")
    string = string.replace(" ! ", "! ")
    string = string.replace(" ? ", "? ")
    string = string.replace(" , ", ", ")
    # double brackets
    string = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", string)
    string = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", string)
    string = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", string)
    string = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', string)
    string = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", string)
    # miscellaneous
    string = string.replace("= = = =", "====")
    string = string.replace("= = =", "===")
    string = string.replace("= =", "==")
    string = string.replace(" " + chr(176) + " ", chr(176))
    string = string.replace(" \n", "\n")
    string = string.replace("\n ", "\n")
    string = string.replace(" N ", " 1 ")
    string = string.replace(" 's", "'s")
    return string

def ptb_detokenizer(x):
    x = x.replace(" 's", "'s")
    x = x.replace("s ' ", "s' ")
    x = x.replace(" n't", "n't")
    x = x.replace(" \n ", "\n")
    x = x.replace("\\/", "/")
    for _ in range(10):
        x = x.replace(" N ", " 1 ")
    x = x.replace("$ 1", "$1")
    x = x.replace("# 1", "#1")
    x = x.replace("<unk>", "?")
    return x

def lm1b_detokenizer(x):
    x = x.replace('http : / / ', 'http://')
    x = x.replace('https : / / ', 'https://')
    x = re.sub(r' \'(\w+)', r"'\1", x)
    x = re.sub(r' (\w+) \. ', r' \1. ', x)
    x = re.sub(r' (\w+) \.$', r' \1.', x)
    x = x.replace(' ? ', '? ')
    x = re.sub(r' \?$', '?', x)
    x = x.replace(' ! ', '! ')
    x = re.sub(r' \!$', '!', x)
    x = x.replace(' , ', ', ')
    x = x.replace(' : ', ': ')
    x = x.replace(' ; ', '; ')
    x = x.replace(' / ', '/')
    x = re.sub(r'\" ([^\"]+) \"', r'"\1"', x)
    x = re.sub(r'\' ([^\']+) \'', r"'\1'", x)
    x = re.sub(r'\( ([^\(\)]+) \)', r"(\1)", x)
    x = re.sub(r'\[ ([^\[\]]+) \]', r"[\1]", x)
    x = x.replace('$ ', '$')
    x = x.replace('£ ', '£')
    return x


def lambada_detokenizer(text):
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    return '\n'+text.strip()


def get_lambada_test_dataset():
    url = "https://openaipublic.blob.core.windows.net/gpt-2/data/lambada_test.jsonl"

    def read_jsonl_to_list(url):
        response = requests.get(url, stream=True)
        data_list = []

        # Process each line in the response content
        for line in response.iter_lines(decode_unicode=True):
            if line:
                data = json.loads(line)
                data_list.append(data)

        return data_list

    lambada_data = read_jsonl_to_list(url)
    dataset = Dataset.from_list(lambada_data)
    return dataset

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


def get_dataloaders(config, distributed=True):
    if config.training.batch_size % (config.ngpus * config.training.accum) != 0:
            raise ValueError(f"Train Batch Size {config.training.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")
    if config.eval.batch_size % (config.ngpus * config.training.accum) != 0:
        raise ValueError(f"Eval Batch Size for {config.eval.batch_size} is not divisible by {config.ngpus} gpus with accumulation {config.training.accum}.")

    train_set = get_dataset(config.data.train_path)
    valid_set = get_dataset(config.data.valid_path)

    if distributed:
        train_sampler = DistributedSequencePackingSampler(train_set,
                                                          max_length=config.model.length, # TODO: make sure this gets the right length with distributed, and make it distribute properly
                                                          total_length=config.model.length * config.training.batch_size // (config.ngpus * config.training.accum), # TODO: make sure this gets the right length with distributed, and make it distribute properly
                                                          drop_last=config.training.drop_last,
        )
        val_sampler = DistributedSequencePackingSampler(valid_set, 
                                                        max_length=config.model.length, # TODO: make sure this gets the right length with distributed, and make it distribute properly
                                                        total_length=config.model.length * config.training.batch_size // (config.ngpus * config.training.accum), # TODO: make sure this gets the right length with distributed, and make it distribute properly
                                                        drop_last=config.training.drop_last,
        )
    else:
        train_sampler = None
        val_sampler = None

    train_loader = cycle_loader(DataLoader(
        train_set,
        # batch_size=config.training.batch_size // (config.ngpus * config.training.accum),
        batch_sampler=train_sampler,
        num_workers=4, # TODO: set up to 8 maybe. 8 GPUs * 2 dataloaders * 8 processes = 128 processes
        collate_fn=train_sampler.collate_fn,
        pin_memory=True,
        shuffle=(train_sampler is None),
        persistent_workers=True,
    ))
    valid_loader = cycle_loader(DataLoader(
        valid_set,
        # batch_size=config.eval.batch_size // (config.ngpus * config.training.accum),
        batch_sampler=val_sampler,
        num_workers=4,
        collate_fn=val_sampler.collate_fn,
        pin_memory=True,
        shuffle=(val_sampler is None),
    ))
    return train_loader, valid_loader


class SequencePackingSampler(Sampler):
    def __init__(self, 
                 dataset,
                 indices=None,
                 max_length=4096, 
                 total_length=4096*32,
                 seed=0,
                 drop_last=False,
    ):
        self.dataset = dataset # a clustered dataset
        self.max_length = max_length
        self.total_length = total_length
        self.seed = seed
        self.epoch = 0
        self.drop_last = drop_last

        if indices is None: #Indices should only be used with ditributed sampler
            self.indices = list(range(len(dataset)))
            self.shuffle = True # only shuffle when we are not using distributed sampler. The distributed sampler should handle the shuffling
        else:
            self.indices = indices
            self.shuffle = False # only shuffle when we are not using distributed sampler. The distributed sampler should handle the shuffling

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.indices), generator=g).tolist()
        else:
            indices = self.indices

        batch = []
        batch_length = 0
        for idx in indices:
            cluster_size = self.dataset[idx]['cluster_size'].item()
            cluster_idx = self.epoch % cluster_size
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
        generator = None # TODO: add generator for deterministic evaluation

        # get the size of the clusters, and use the epoch to index into the cluster
        cluster_sizes = [x['cluster_size'] for x in batch]
        indexes = [self.epoch % cs for cs in cluster_sizes]

        # get the input_ids for each cluster
        input_ids_list = [x["input_ids"][i] for x, i in zip(batch, indexes)]
        input_ids_list = [maybe_truncate(input_ids, self.max_length, generator=generator) for input_ids in input_ids_list[:-1]]

        # the last input ids should be truncated to the remaining length, to not exceed the total length
        length_sum = sum([min(x["length"][i], self.max_length) for x, i in zip(batch, indexes[:-1])]).item()
        last_length = min(self.total_length - length_sum, self.max_length)
        print("total length: ", length_sum + last_length)

        input_ids_list.append(maybe_truncate(input_ids_list[-1], last_length, generator=generator))

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
                 seed=0, 
                 max_length=4096, # max length of each sequence in the batch
                 total_length=4096*32, # max length of the batch (amount of tokens in the batch)
                 drop_last=False,
    ):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=False)

        self.max_length = max_length
        self.total_length = total_length
        self.seed = seed
        self.dataset = dataset
        self.drop_last = drop_last

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
