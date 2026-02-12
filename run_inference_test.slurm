#!/bin/bash
#SBATCH --job-name=ldm_inf_single
#SBATCH --partition=coe-gpu
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --output=logs/inference_single_%j.out

# Usage: sbatch run_inference_test.slurm <CONFIG_PATH> <CKPT_PATH> <DATA_PATH>

if [ "$#" -ne 3 ]; then
    echo "Usage: sbatch run_inference_test.slurm <CONFIG_PATH> <CKPT_PATH> <DATA_PATH>"
    exit 1
fi

CONFIG=$1
CKPT=$2
DATA=$3

echo "Job ID: $SLURM_JOB_ID"
echo "Config: $CONFIG"
echo "Checkpoint: $CKPT"
echo "Data Path: $DATA"

module load cuda/11.8
source $HOME/.bashrc
conda activate ldm

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
# python inference_test.py sets specific GPU if needed, but usually CUDA_VISIBLE_DEVICES is handled by SLURM
export PYTHONUNBUFFERED=1

echo "Starting inference..."
# Create specific output directory for this run to avoid collisions
MODEL_NAME=$(basename $(dirname $(dirname $CKPT))) # logs/DATE_NAME/checkpoints/file.ckpt -> DATE_NAME
OUTDIR="inference_results_slurm/${MODEL_NAME}"

echo "Output Directory: $OUTDIR"
mkdir -p $OUTDIR

python inference_test.py \
    --config "$CONFIG" \
    --ckpt "$CKPT" \
    --data_path "$DATA" \
    --outdir "$OUTDIR" \
    --num_samples 10

echo "Inference finished."
