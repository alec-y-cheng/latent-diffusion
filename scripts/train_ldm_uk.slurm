#!/bin/bash
#SBATCH --job-name=ldm_uk
#SBATCH --partition=gpu-h200
##SBATCH --qos=coe-ice
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=logs/train_%j.out

#SBATCH --account=gts-pkastner3
#SBATCH --qos=embers

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

# UK dataset is a single HDF5 file. Use the stable project path by default:
# job-local /tmp has been observed to be cleaned while long runs are active,
# which makes newly spawned DataLoader workers fail when they reopen the HDF5.
DATA_FILE=${DATA_FILE:-data/uk_roof_dataset.h5}
if [[ "$DATA_FILE" != /* ]]; then
  DATA_FILE="$PROJECT_ROOT/$DATA_FILE"
fi
if [[ ! -f "$DATA_FILE" ]]; then
  echo "ERROR: dataset file not found: $DATA_FILE" >&2
  exit 3
fi

TRAIN_DATA_FILE="$DATA_FILE"
if [[ "${LDM_STAGE_DATA:-0}" == "1" ]]; then
  stage_file_to_tmp "$DATA_FILE"
  TRAIN_DATA_FILE="$STAGED_DATA_FILE"
  echo "Using staged dataset: $TRAIN_DATA_FILE"
else
  echo "Using stable dataset path: $TRAIN_DATA_FILE"
  echo "Set LDM_STAGE_DATA=1 to copy the dataset to job-local tmp."
fi

python main.py \
  -t \
  --gpus 0, \
  ${EXTRA_ARGS:-} \
  data.params.train.params.data_path="$TRAIN_DATA_FILE" \
  data.params.validation.params.data_path="$TRAIN_DATA_FILE"
