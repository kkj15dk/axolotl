#!/usr/bin/bash
#SBATCH --job-name train_Axolotl
#SBATCH --account eu-25-27
#SBATCH --partition qgpu
#SBATCH --gpus 8
#SBATCH --nodes 1
#SBATCH --time 48:00:00

ml purge
ml Python/3.13.1-GCCcore-14.2.0
source .venv/bin/activate

# here follow the commands you want to execute

# NCCL diagnostics and safer async error handling
export NCCL_DEBUG=INFO
# export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
# export TORCH_NCCL_BLOCKING_WAIT=1
# Optional deeper tracing (can be verbose)
# export TORCH_NCCL_TRACE_BUFFER_SIZE=1048576
# Optional CUDA sync for debugging kernel errors (slower)
# export CUDA_LAUNCH_BLOCKING=1

python3 axolotl/train.py # load_dir=/scratch/project/eu-25-27/exp_local/UniRef50_unclustered/2025.08.22/113333