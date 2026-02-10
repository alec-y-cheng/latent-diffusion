#!/bin/bash
#SBATCH --job-name=ldm_cfd
#SBATCH --partition=coe-gpu
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=012:00:00
#SBATCH --output=logs/train_%j.out

module load cuda/11.8
source $HOME/.bashrc
conda activate ldm

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0 

$CONDA_PREFIX/bin/python main.py \
  --base configs/latent-diffusion/cfd_ldm.yaml \
  -t \
  --gpus 0, \
  #--resume logs/2026-02-02T01-45-25_cfd_ldm/checkpoints/last.ckpt \
  "$@"