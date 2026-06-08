#!/bin/bash
#SBATCH --job-name=aug_consistency
#SBATCH --account=gts-pkastner3
#SBATCH --partition=gpu-h200
#SBATCH --qos=inferno
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=08:00:00
#SBATCH --output=logs/augmentation_consistency_%j.out

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

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

# MAX_SAMPLES=0 means use the entire split. For a quick smoke test:
#   MAX_SAMPLES=20 NUM_EXAMPLES=4 sbatch scripts/augmentation-check/run_augmentation_consistency.slurm
python scripts/augmentation-check/augmentation_consistency.py \
    --logs "${LOGS:-logs}" \
    --filter "${FILTER:-uk_grad_corr_low,uk_pinns_baseline}" \
    --data_path "${DATA_PATH:-data/uk_roof_dataset.h5}" \
    --split "${SPLIT:-train}" \
    --steps "${STEPS:-50}" \
    --max_samples "${MAX_SAMPLES:-0}" \
    --num_examples "${NUM_EXAMPLES:-10}" \
    --seed "${SEED:-23}" \
    --transforms "${TRANSFORMS:-hflip,vflip,rot180,transpose,rot90,rot270,anti_transpose}" \
    --outdir_suffix "${OUTDIR_SUFFIX:-augmentation_consistency}"

echo "Augmentation consistency check finished."
