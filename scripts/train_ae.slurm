#!/bin/bash
#SBATCH --job-name=ae_cfd
#SBATCH --partition=coe-gpu
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=012:00:00
#SBATCH --output=logs/train_%j.out

module load cuda/11.8
source $HOME/.bashrc
conda activate ldm

export CUDA_VISIBLE_DEVICES=0 

python main.py \
  --base configs/autoencoder/autoencoder_kl_32x32x4_uk.yaml \
  -t \
  --gpus 0, \
  --resume_from_checkpoint logs/2026-04-27T15-01-16_autoencoder_kl_32x32x4_uk/checkpoints/last.ckpt