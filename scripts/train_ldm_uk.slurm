#!/bin/bash
#SBATCH --job-name=ldm_uk
##SBATCH --partition=coe-gpu
##SBATCH --qos=coe-ice
##SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out

#SBATCH --account=gts-pkastner3
#SBATCH --qos=inferno
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

# UK dataset is a single HDF5 file — copy it to local NVMe for fast IO
DATA_FILE=${DATA_FILE:-data/uk_roof_dataset.h5}
stage_file_to_tmp "$DATA_FILE"

python main.py \
  -t \
  --gpus 0, \
  ${EXTRA_ARGS:-} \
  data.params.train.params.data_path="$STAGED_DATA_FILE" \
  data.params.validation.params.data_path="$STAGED_DATA_FILE"
