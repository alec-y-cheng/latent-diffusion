#!/bin/bash
#SBATCH --job-name=ae_cfd
##SBATCH --partition=coe-gpu
##SBATCH --qos=coe-ice
##SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
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

export TMPDIR="${TMPDIR:-/tmp/${USER}/${SLURM_JOB_ID:-$$}}"
mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=0 

DATA_FILE=${DATA_FILE:-data/uk_roof_dataset.h5}
if [[ "${STAGE_DATA:-0}" == "1" ]]; then
  stage_file_to_tmp "$DATA_FILE"
  RUN_DATA_FILE="$STAGED_DATA_FILE"
else
  if [[ ! -f "$DATA_FILE" ]]; then
    echo "ERROR: dataset file not found: $DATA_FILE" >&2
    exit 3
  fi
  RUN_DATA_FILE="$DATA_FILE"
fi

NUM_WORKERS=${LDM_NUM_WORKERS:-2}

python main.py \
  --base configs/autoencoder/autoencoder_kl_32x32x4_uk.yaml \
  -t \
  --gpus 0, \
  ${EXTRA_ARGS:-} \
  data.params.num_workers="$NUM_WORKERS" \
  data.params.train.params.data_path="$RUN_DATA_FILE" \
  data.params.validation.params.data_path="$RUN_DATA_FILE"
