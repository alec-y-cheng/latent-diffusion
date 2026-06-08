#!/bin/bash
#SBATCH --job-name=check_norm
#SBATCH --account=gts-pkastner3
#SBATCH --qos=inferno
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/check_normalization_%j.out

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/scripts/slurm_env.sh" ]]; then
  PROJECT_ROOT="$SLURM_SUBMIT_DIR"
else
  PROJECT_ROOT="/storage/home/hcoda1/7/acheng324/scratch/latent-diffusion"
fi
cd "$PROJECT_ROOT"

echo "Job ID: ${SLURM_JOB_ID:-manual}"
echo "Project root: $PROJECT_ROOT"

module load cuda/11.8
source scripts/slurm_env.sh
activate_ldm_env

python scripts/check_dataset_normalization.py \
  --data_path "${DATA_PATH:-data/uk_roof_dataset.h5}" \
  --limit "${LIMIT:-500}" \
  --chunk_size "${CHUNK_SIZE:-16}"
