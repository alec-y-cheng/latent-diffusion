#!/bin/bash
#SBATCH --job-name=ae_cfd
#SBATCH --partition=coe-gpu
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out

module load cuda/11.8
source activate ldm

export CUDA_VISIBLE_DEVICES=0 

python main.py \
  --base configs/autoencoder/autoencoder_kl_32x32x4.yaml \
  -t \
  --gpus 0,
