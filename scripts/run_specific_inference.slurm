#!/bin/bash
#SBATCH --job-name=ldm_inf_specific
#SBATCH --partition=coe-gpu
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/inference_specific_%j.out

# Usage: sbatch scripts/run_specific_inference.slurm

echo "Job ID: $SLURM_JOB_ID"

module load cuda/11.8
source $HOME/.bashrc
conda activate ldm

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

echo "Starting Fast Batch Inference for specific models..."

# Run the optimized fast batch script targeting specific folders
# Change the --filter argument to include any comma-separated folders you want
python scripts/fast_batch_inference.py \
    --logs logs \
    --num_samples 50 \
    --steps 50 \
    --filter cfd_ldm_wavelet

echo "Batch Inference finished."
