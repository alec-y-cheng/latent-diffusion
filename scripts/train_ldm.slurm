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

export CUDA_VISIBLE_DEVICES=0 

# Cache the massive dataset to the node's local NVMe drive to prevent network IO bottlenecks
echo "Copying dataset to local NVMe ($TMPDIR)..."
mkdir -p $TMPDIR/data/augmented
cp data/augmented/full_dataset.npz $TMPDIR/data/augmented/
echo "Copy complete."

python main.py \
  --base configs/latent-diffusion/cfd_ldm.yaml \
  -t \
  --gpus 0, \
  $EXTRA_ARGS \
  data.params.train.params.data_path=$TMPDIR/data/augmented \
  data.params.validation.params.data_path=$TMPDIR/data/augmented
  #--resume logs/2026-02-02T01-45-25_cfd_ldm/checkpoints/epoch=000061.ckpt