# %%
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from transformers import PreTrainedTokenizerFast
import argparse


# %%
# Define the transformation function for batches
def preprocess(example: dict,
               tokenizer: PreTrainedTokenizerFast,
               sequence_key: str = 'sequence', 
               label_key: str = 'domainid', 
):
    sequence = example[sequence_key]
    label = example[label_key]
    assert isinstance(sequence, str), f"sequence is not a string: {sequence}"
    assert isinstance(label, int), f"label is not an int: {label}"

    if label == 2: # Prokaryotic
        label = 0
    elif label == 2759: # Eukaryotic
        label = 1
    else:
        raise ValueError(f"Invalid label: {label}")
    
    input_ids = tokenizer(sequence,
                          return_tensors='pt',
    )['input_ids'].squeeze(0) # My tokenizer adds BOS and EOS tokens
    length = len(input_ids)

    return {'cluster_size': 1,'input_ids': input_ids, 'label': label, 'length': length}


def stream_groupby_gen(dataset: Dataset, 
                       id_key: str, 
                       chunk_size=10000, 
):
    '''
    Input:
    A dataset with columns 'input_ids', 'label', and id_key. id_key is the column to group by, and will be renamed to 'id'.
    '''
    agg = lambda chunk: chunk.groupby('id').agg({
        'cluster_size': 'sum',
        'label': list, 
        'length': list,
        'input_ids': list,
        })

    # Tell pandas to read the data in chunks
    chunks = dataset.rename_column(id_key, 'id').select_columns(['id', 'cluster_size', 'label', 'input_ids', 'length']).to_pandas(batched=True, batch_size=chunk_size)
    
    orphans = pd.DataFrame()

    for chunk in tqdm(chunks, desc='Processing chunks', unit='chunk', total=len(dataset)//chunk_size):

        # Add the previous orphans to the chunk
        chunk = pd.concat((orphans, chunk))

        # Determine which rows are orphans
        last_val = chunk['id'].iloc[-1]
        is_orphan = chunk['id'] == last_val

        # Put the new orphans aside
        chunk, orphans = chunk[~is_orphan], chunk[is_orphan]
        # Perform the aggregation and store the results
        chunk = agg(chunk).reset_index()

        dataset = Dataset.from_pandas(chunk)

        for i in range(len(chunk)):
            yield dataset[i]

    # Don't forget the remaining orphans
    if len(orphans):
        chunk = agg(orphans).reset_index()

        dataset = Dataset.from_pandas(chunk)

        for i in range(len(chunk)):
            yield dataset[i]

# %%
# Encode the dataset
def main():

    parser = argparse.ArgumentParser(description="Convert csv to nested dataset")
    parser.add_argument("--input_path", default='datasets/test_sorted.csv', type=str)
    parser.add_argument("--output_path", default='datasets/', type=str)
    parser.add_argument("--filename_encoded", default='test_encoded', type=str)
    parser.add_argument("--filename_grouped", default='test_grouped', type=str)
    parser.add_argument("--sequence_key", default='sequence', type=str)
    parser.add_argument("--id_key", default='clusterid', type=str)
    parser.add_argument("--label_key", default='domainid', type=str)
    parser.add_argument("--train_test_split_ratio", default=0.02, type=float)
    parser.add_argument("--val_test_split_ratio", default=0.5, type=float)
    parser.add_argument("--num_proc", default=os.cpu_count(), type=int)
    parser.add_argument("--tokenizer_path", default='/home/kkj/axolotl/tokenizer/tokenizer_uniform', type=str)
    parser.add_argument("--cache_dir", default=None, type=str)
    args = parser.parse_args()

    input_path: str = args.input_path
    output_path: str = args.output_path
    filename_encoded: str = args.filename_encoded
    filename_grouped: str = args.filename_grouped
    sequence_key: str = args.sequence_key
    id_key: str = args.id_key
    label_key: str = args.label_key
    train_test_split_ratio: float = args.train_test_split_ratio
    val_test_split_ratio: float = args.val_test_split_ratio
    num_proc: int = args.num_proc
    tokenizer_path: str = args.tokenizer_path
    cache_dir: str = args.cache_dir

    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    assert label_key != 'label', "label_key cannot be 'label', as it is used as a temporary column name, and deleted afterwards"
    assert sequence_key != 'input_ids', "sequence_key cannot be 'input_ids', as it is used as a temporary column name, and deleted afterwards"

    if not os.path.exists(f'{output_path}{filename_encoded}'):
        # Load the dataset
        print(f"Loading {input_path}")
        dataset = (
            load_dataset('csv', data_files=input_path, cache_dir=cache_dir)['train']
            # .rename_column(' domainid', 'domain')
            # .rename_column(' sequence', 'sequence')
            # .rename_column(' cluster90id', 'cluster90id')
            # .rename_column(' cluster100id', 'cluster100id')
        )

        print(f"Loaded {input_path} with headers:")
        print(dataset.column_names)
        print(f"Dataset length: {len(dataset)}")
        print(f"Encoding {filename_encoded}")
        dataset = dataset.map(preprocess, 
                    fn_kwargs={
                        'tokenizer': tokenizer,
                        'sequence_key': sequence_key, 
                        'label_key': label_key,
                    },
                    remove_columns=[label_key, sequence_key],
                    batched=False, 
                    num_proc=num_proc,
        )
        dataset.save_to_disk(f'{output_path}{filename_encoded}')
    else:
        print(f"{filename_encoded} already exsists, loading it")
        dataset = load_from_disk(f'{output_path}{filename_encoded}')
        print("Loaded dataset, starting grouping")

    # %%
    # Group by the id column and aggregate the input_ids, labels, and lengths
    if not os.path.exists(f'{output_path}{filename_grouped}'):
        print(f"Grouping {filename_grouped}")
        dataset = Dataset.from_generator(stream_groupby_gen, 
                                        gen_kwargs={'dataset': dataset, 'id_key': id_key},
                                        cache_dir=cache_dir,
        ) # .with_format('numpy')
        print("Grouping done, splitting into train/test/val, and then saving to disk")
        # Split the dataset into train and temp sets using the datasets library
        train_val_test_split = dataset.train_test_split(test_size=train_test_split_ratio, seed=42)
        train_dataset = train_val_test_split['train']
        temp_dataset = train_val_test_split['test']

        # Split the temp set into validation and test sets using the datasets library
        val_test_split = temp_dataset.train_test_split(test_size=val_test_split_ratio, seed=42)
        val_dataset = val_test_split['train']
        test_dataset = val_test_split['test']

        dataset = DatasetDict({
            'train': train_dataset,
            'valid': val_dataset,
            'test': test_dataset,
        })

        dataset.save_to_disk(f'{output_path}{filename_grouped}')
    else:
        print(f"{filename_grouped} already exisits, loading it")
        dataset = load_from_disk(f'{output_path}{filename_grouped}')


    print('Doen')
    print(dataset)

if __name__ == '__main__':
    main()