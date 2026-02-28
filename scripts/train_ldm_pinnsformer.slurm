#!/bin/bash
#SBATCH --job-name=ldm_cfd_pinns
#SBATCH --partition=coe-gpu
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_pinns_%j.out

module load cuda/11.8
source $HOME/.bashrc
conda activate ldm

export CUDA_VISIBLE_DEVICES=0 

python main.py \
  --base configs/latent-diffusion/cfd_ldm_pinnsformer.yaml \
  -t \
  --gpus 0, \
  $EXTRA_ARGS
