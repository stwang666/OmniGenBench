#!/bin/bash
#SBATCH --job-name=biclass_train
#SBATCH --output=biclass_training_%j.log
#SBATCH --gpus=2
#SBATCH --time=12:00:00


python biclass_seq_tissue_repair_forward.py
