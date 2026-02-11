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
echo "Script arguments: $@"

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0 
export PYTHONUNBUFFERED=1

# Construct arguments
ARGS="--base configs/latent-diffusion/cfd_ldm.yaml -t --gpus 0,"

if [ ! -z "$RESUME" ]; then
  echo "Resuming from: $RESUME"
  ARGS="$ARGS --resume $RESUME"
fi

if [ ! -z "$EXTRA_ARGS" ]; then
  echo "Applying extra arguments: $EXTRA_ARGS"
  ARGS="$ARGS $EXTRA_ARGS"
fi

echo "Running command: $CONDA_PREFIX/bin/python main.py $ARGS $@"

$CONDA_PREFIX/bin/python main.py $ARGS "$@"