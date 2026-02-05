#!/bin/bash
#SBATCH --job-name=eval_cfd
#SBATCH --partition=coe-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_%j.out

source $HOME/.bashrc
conda activate ldm
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Update these paths!
CONFIG="configs/latent-diffusion/cfd_ldm.yaml"
CKPT="logs/2026-02-02T01-45-25_cfd_ldm/checkpoints/last.ckpt"

$CONDA_PREFIX/bin/python eval_metrics.py \
  --config $CONFIG \
  --ckpt $CKPT \
  --outdir eval_results \
  --num_samples 200
