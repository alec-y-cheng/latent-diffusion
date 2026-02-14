#!/bin/bash
#SBATCH --job-name=verify_channels
#SBATCH --partition=coe-gpu
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/verify_%j.out

module load cuda/11.8
source $HOME/.bashrc
conda activate ldm

echo "Starting verification..."
echo "Running on node: $SLURMD_NODENAME"

# Run the verification script
# Default data path
DATA_PATH="data/full_dataset.npz"

# Allow overriding via command line
if [ ! -z "$1" ]; then
    DATA_PATH="$1"
fi

python verify_channels.py "$DATA_PATH"
