#!/bin/bash
#SBATCH --job-name=ldm_inf_single
#SBATCH --partition=coe-gpu
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=logs/inference_single_%j.out

# Usage: sbatch scripts/run_inference_test.slurm

# No arguments needed, it scans the 'logs' folder automatically

echo "Job ID: $SLURM_JOB_ID"

module load cuda/11.8
source $HOME/.bashrc
conda activate ldm

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

echo "Starting Batch Inference on All Latest Checkpoints..."

# Run the master batch script
# This will:
# 1. Scan logs/
# 2. Find latest experiments
# 3. Run inference on each (validation set)
# 4. Generate 'all_experiments_summary.csv'

# Run the optimized fast batch script
# This loads the dataset ONCE and iterates models in the same process
# avoiding 35-minute startup times per model.

python scripts/fast_batch_inference.py \
    --logs logs \
    --num_samples 50 \
    --steps 50

echo "Batch Inference finished."
