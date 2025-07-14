# %%
import os
import numpy as np
import torch.distributed as dist
import datetime
import torch
import torch.multiprocessing as mp
import time
import sys

import axolotl.data_nested as data

def setup_distributed():
    """Setup environment variables for distributed training"""
    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

def run_multiprocess_test(rank, world_size, test_batches=500, test_every_batches=50):
    """Initialize distributed process group and run test"""
    try:
        # Initialize the process group for distributed training
        dist.init_process_group(backend='gloo', rank=rank, world_size=world_size)
        print(f"Rank {rank}/{world_size-1}: Process group initialized")
        
        # Run the main function for each process
        run(rank, world_size, test_batches, test_every_batches)
        
    except Exception as e:
        print(f"Rank {rank}: Error occurred: {e}")
        raise
    finally:
        # Clean up the process group
        if dist.is_initialized():
            dist.destroy_process_group()
            print(f"Rank {rank}: Process group destroyed")

def run(rank, world_size, test_batches=500, test_every_batches=50):
    """Main test function for each rank"""
    BS = 4
    CTX_LEN = 512
    expected_tokens_per_batch = BS * CTX_LEN

    print(f"Rank {rank}: Starting dataloader test with BS={BS}, CTX_LEN={CTX_LEN}")
    
    # Build data iterators
    train_ds, eval_ds = data.get_dataloaders(BS,
                                            BS,
                                            world_size,
                                            1,
                                            '/home/kkj/axolotl/datasets/IPR036736_90_grouped/train',
                                            '/home/kkj/axolotl/datasets/IPR036736_90_grouped/valid',
                                            CTX_LEN,
                                            drop_last=True,
                                            num_workers=0,  # Keep 0 for distributed testing to avoid multiprocessing conflicts
                                            distributed=True,
                                            epoch=0,
    )
    
    print(f"Rank {rank}: Dataloaders created successfully")
    
    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    # Test both train and eval iterators
    for iterator_name, iterator in [("eval", eval_iter), ("train", train_iter)]:
        print(f"Rank {rank}: Testing {iterator_name} iterator")
        
        batch_count = 0
        total_tokens_processed = 0
        start_time = time.time()
        
        try:
            for i in range(test_batches):
                batch = next(iterator)
                batch_count += 1
                
                # Get the nested tensor
                input_ids = batch['input_ids']
                
                # Print total number of tokens
                total_tokens = input_ids.numel()
                total_tokens_processed += total_tokens
                
                # Check token count expectations
                if total_tokens > expected_tokens_per_batch:
                    print(f"Rank {rank}: Warning: Batch {i} has {total_tokens} tokens, exceeds expected {expected_tokens_per_batch}")
                
                # Detailed logging every 100 batches
                if i % 100 == 0:
                    elapsed = time.time() - start_time
                    tokens_per_sec = total_tokens_processed / elapsed if elapsed > 0 else 0
                    
                    print(f"Rank {rank}: {iterator_name.upper()} Batch {i}")
                    print(f"  Total tokens: {total_tokens}")
                    print(f"  Batch size (sequences): {input_ids.size(0)}")
                    print(f"  Individual sequence lengths: {[len(seq) for seq in input_ids]}")
                    print(f"  Cumulative tokens processed: {total_tokens_processed}")
                    print(f"  Tokens/sec: {tokens_per_sec:.2f}")
                    print(f"  Elapsed time: {elapsed:.2f}s")
                    print("-" * 50)
                
                # Test epoch cycling every so often
                if i > 0 and i % test_every_batches == 0:
                    # Access epoch through the CyclingDataLoader
                    if iterator_name == "train":
                        current_epoch = train_ds.epoch
                        print(f"Rank {rank}: Current epoch: {current_epoch}")
                    elif iterator_name == "eval":
                        current_epoch = eval_ds.epoch
                        print(f"Rank {rank}: Current epoch: {current_epoch}")
                        
        except StopIteration:
            print(f"Rank {rank}: {iterator_name.upper()} iterator exhausted after {batch_count} batches")
        except Exception as e:
            print(f"Rank {rank}: Error in {iterator_name} iterator at batch {batch_count}: {e}")
            raise
        
        elapsed_total = time.time() - start_time
        avg_tokens_per_sec = total_tokens_processed / elapsed_total if elapsed_total > 0 else 0
        
        print(f"Rank {rank}: {iterator_name.upper()} test completed:")
        print(f"  Batches processed: {batch_count}")
        print(f"  Total tokens: {total_tokens_processed}")
        print(f"  Average tokens/sec: {avg_tokens_per_sec:.2f}")
        print(f"  Total time: {elapsed_total:.2f}s")
        print("=" * 60)

def main():
    """Main function to spawn multiprocess distributed test"""
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    
    # Setup distributed environment
    setup_distributed()
    
    # Configuration
    world_size = 2  # Number of processes (simulating 2 GPUs)
    test_batches = 5000  # Number of batches to test per iterator
    test_every_batches = 200  # Frequency to test epoch cycling
    
    print(f"Starting multiprocess distributed dataloader test")
    print(f"World size: {world_size}")
    print(f"Test batches per iterator: {test_batches}")
    print("=" * 60)
    
    try:
        # Spawn processes
        processes = []
        for rank in range(world_size):
            p = mp.Process(target=run_multiprocess_test, args=(rank, world_size, test_batches, test_every_batches))
            p.start()
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Check if all processes completed successfully
        all_success = all(p.exitcode == 0 for p in processes)
        
        if all_success:
            print("=" * 60)
            print("✓ All processes completed successfully!")
            print("✓ Distributed dataloader test PASSED")
        else:
            print("=" * 60)
            print("✗ Some processes failed!")
            for i, p in enumerate(processes):
                print(f"  Process {i} exit code: {p.exitcode}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        # Terminate all processes
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()
        sys.exit(1)
    except Exception as e:
        print(f"Error in main: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
# %%
