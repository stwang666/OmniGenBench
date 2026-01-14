#!/bin/bash
#SBATCH --job-name=triclass_train
#SBATCH --output=training_%j.log
#SBATCH --gpus=2
#SBATCH --time=24:00:00


python triclass_seq_tissue.py
