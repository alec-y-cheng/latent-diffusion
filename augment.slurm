#!/bin/bash
#SBATCH --job-name=augment_data
#SBATCH --partition=coe-gpu
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/augment.out

source $HOME/.bashrc
conda activate ldm

# Run augmentation (Input File -> Output Folder)
python data_augmentation.py --input data/full_dataset.npz --output data/augmented
