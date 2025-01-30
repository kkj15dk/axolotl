#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
###BSUB -R "select[gpu32gb]"
### -- set the job Name -- 
#BSUB -J convert_csv
### -- ask for number of cores (default: 4) -- 
#BSUB -n 16
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify amount of memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"
### -- set walltime limit: hh:mm -- 
#BSUB -W 24:00
### -- send notification at start -- 
#BSUB -B
### -- send notification at completion -- 
#BSUB -N
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Logs/%J.out 
#BSUB -e Logs/%J.err 

module load python3/3.13.0
source .venv/bin/activate

# here follow the commands you want to execute
python3 axolotl/data_utils/convert_csv_to_nested_dataset.py