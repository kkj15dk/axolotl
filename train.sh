#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpua100
#BSUB -R "select[gpu80gb]"
### -- if your job needs to depend on another job --
###BSUB -w "done(23988842)"
### -- set the job Name -- 
#BSUB -J train_DiT
### -- ask for number of cores (default: 4) -- 
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
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
python3 axolotl/train.py