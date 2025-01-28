# %%
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from transformers import PreTrainedTokenizerFast

# %%
# Define the parameters
sequence_key = 'sequence'
id_key = 'clusterid' # This is the column to group by
label_key = 'domainid' # This is the column to use as the label
output_path = '/home/kkj/axolotl/datasets/'
# input_path = '/home/kkj/ProtDiffusion/datasets/UniRef50_sorted.csv' # Has to be sorted by id
# input_path = '/home/kkj/ProtDiffusion/datasets/UniRefALL_sorted.csv'
input_path = '/home/kkj/axolotl/datasets/90_IPR036736_sorted.csv'
filename_encoded = 'IPR036736_90'
filename_grouped = 'IPR036736_90'
filename_nested = 'IPR036736_90'
assert label_key != 'label', "label_key cannot be 'label', as it is used as a temporary column name, and deleted afterwards"

tokenizer = PreTrainedTokenizerFast.from_pretrained('/home/kkj/axolotl/tokenizer/tokenizer_uniform')

# %%
# Define the transformation function for batches
def preprocess(example: dict,
               sequence_key: str = 'sequence', 
               label_key: str = 'domainid', 
):
    sequence = example[sequence_key]
    label = example[label_key]
    if label == 2: # Prokaryotic
        label = 0
    elif label == 2759: # Eukaryotic
        label = 1
    else:
        raise ValueError(f"Invalid label: {label}")
    
    input_ids = tokenizer(sequence,
                          return_tensors='pt',
    )['input_ids'].squeeze(0)
    
    return {'input_ids': input_ids, 'label': label}


def stream_groupby_gen(dataset: Dataset, 
                       id_key: str, 
                       chunk_size=10000, 
):
    '''
    Input:
    A dataset with columns 'input_ids', 'label', and id_key. id_key is the column to group by, and will be renamed to 'id'.
    '''
    agg = lambda chunk: chunk.groupby('id').agg({
        'label': list, 
        'input_ids': list
        })

    # Tell pandas to read the data in chunks
    chunks = dataset.rename_column(id_key, 'id').select_columns(['id','label','input_ids']).to_pandas(batched=True, batch_size=chunk_size)
    
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
if not os.path.exists(f'{output_path}{filename_encoded}'):
    # Load the dataset
    print(f"Loading {input_path}")
    dataset = load_dataset('csv', data_files=input_path)['train']
    print(f"Loaded {input_path}")
    print(f"Dataset length: {len(dataset)}")
    print(f"Encoding {filename_encoded}")
    dataset.map(preprocess, 
                fn_kwargs={'sequence_key': sequence_key, 
                            'label_key': label_key,
                },
                remove_columns=[label_key, sequence_key],
                batched=False, 
                num_proc=16,
    ).save_to_disk(f'{output_path}{filename_encoded}')
else:
    print(f"{filename_encoded} already encoded")

# %%
# Group by the id column and aggregate the input_ids, labels, and lengths
if not os.path.exists(f'{output_path}{filename_grouped}_grouped'):
    print(f"Grouping {filename_grouped}")
    dataset = load_from_disk(f'{output_path}{filename_encoded}')
    print("Loaded dataset, starting grouping")
    dataset = Dataset.from_generator(stream_groupby_gen, 
                                     gen_kwargs={'dataset': dataset, 'id_key': id_key},
    ) # .with_format('numpy')
    print("Grouping done, splitting into train/test/val, and then saving to disk")
    # Split the dataset into train and temp sets using the datasets library
    train_test_split_ratio = 0.02
    train_val_test_split = dataset.train_test_split(test_size=train_test_split_ratio, seed=42)
    train_dataset = train_val_test_split['train']
    temp_dataset = train_val_test_split['test']

    # Split the temp set into validation and test sets using the datasets library
    val_test_split_ratio = 0.5
    val_test_split = temp_dataset.train_test_split(test_size=val_test_split_ratio, seed=42)
    val_dataset = val_test_split['train']
    test_dataset = val_test_split['test']

    dataset_dict = DatasetDict({
        'train': train_dataset,
        'valid': val_dataset,
        'test': test_dataset,
    })

    dataset_dict.save_to_disk(f'{output_path}{filename_grouped}_grouped')
else:
    print(f"{filename_grouped} already grouped")


print('Doen')

# %%
# Load the grouped dataset
dataset = load_from_disk(f'{output_path}{filename_grouped}_grouped')
print(dataset)
print(dataset['train'][0])