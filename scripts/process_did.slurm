#!/bin/bash
#SBATCH --job-name=process_did
#SBATCH --partition=coe-gpu
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/process_did_%j.out

module load cuda/11.8
source $HOME/.bashrc
conda activate ldm

echo "Starting Directional Integrated Distance processing..."
python add_did_to_dataset.py
echo "Finished!"
