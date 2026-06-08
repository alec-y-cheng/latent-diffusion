#!/bin/bash
#SBATCH --job-name=ldm_cfd
##SBATCH --partition=coe-gpu
##SBATCH --qos=coe-ice
##SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=012:00:00
#SBATCH --output=logs/train_%j.out

#SBATCH --account=gts-pkastner3
#SBATCH --qos=embers
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$PROJECT_ROOT" || ! -f "$PROJECT_ROOT/main.py" ]]; then
  PROJECT_ROOT="/storage/scratch1/7/acheng324/latent-diffusion"
fi
cd "$PROJECT_ROOT"

module load cuda/11.8
source "$PROJECT_ROOT/scripts/slurm_env.sh"
activate_ldm_env

export CUDA_VISIBLE_DEVICES=0 

# Cache the massive dataset to the node's local NVMe drive to prevent network IO bottlenecks
# Default to data/augmented if DATA_DIR is not set for backwards compatibility
DATA_DIR=${DATA_DIR:-data/augmented}
stage_dir_to_tmp "$DATA_DIR"

python main.py \
  --base configs/latent-diffusion/cfd_ldm.yaml \
  -t \
  --gpus 0, \
  ${EXTRA_ARGS:-} \
  data.params.train.params.data_path="$STAGED_DATA_DIR" \
  data.params.validation.params.data_path="$STAGED_DATA_DIR"
  #--resume logs/2026-02-02T01-45-25_cfd_ldm/checkpoints/epoch=000061.ckpt
