#!/bin/bash
#SBATCH --job-name=ldm_inference
#SBATCH --partition=coe-gpu
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=logs/inference_%j.out

module load cuda/11.8
source $HOME/.bashrc
conda activate ldm

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0 
export PYTHONUNBUFFERED=1

echo "Starting inference on all checkpoints..."
python run_all_checkpoints.py --logdir logs --outdir inference_results_slurm --steps 50 --num_samples 5

echo "Inference job completed."
