#!/usr/bin/bash
#SBATCH --job-name train_Axolotl
#SBATCH --account eu-25-27
#SBATCH --partition qgpu
#SBATCH --gpus 8
#SBATCH --nodes 1
#SBATCH --time 24:00:00

ml purge
ml Python/3.13.1-GCCcore-14.2.0
source .venv/bin/activate

# here follow the commands you want to execute
python3 axolotl/train.py