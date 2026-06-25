#!/bin/bash
#SBATCH --job-name=ldm_inf_specific
#SBATCH --account=gts-pkastner3
#SBATCH --partition=gpu-h200
#SBATCH --qos=embers
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/inference_specific_%j.out

# Usage: sbatch scripts/run_specific_inference.slurm

echo "Job ID: $SLURM_JOB_ID"

module load cuda/11.8
source scripts/slurm_env.sh
activate_ldm_env

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

echo "Starting Fast Batch Inference for specific models..."

# Run the optimized fast batch script targeting specific folders
# Change the --filter argument to include any comma-separated folders you want
# NOTE: NO SPACES after commas in --filter or bash splits them into positional args



# DID (15-channel) models — must use augmented_did_h5 dataset
 <<'COMMENT'
python scripts/fast_batch_inference.py \
    --logs logs \
    --num_samples 10 \
    --steps 50 \
    --data_path data/augmented_did_h5 \
    --filter pinns_baseline_wavelet_did,physics_hybrid_master_did

echo "Batch Inference finished."

# Standard (8-channel) models

python scripts/fast_batch_inference.py \
    --logs logs \
    --num_samples 10 \
    --steps 50 \
    --data_path data/augmented \
    --filter grad_corr_low_highlr,grad_corr_low_noclip,pinns_baseline_base,grad_corr_low_real,grad_corr_med_real,grad_corr_high_real,pinns_baseline_wavelet,pinns_bc_heavy,wavelet_grad_corr,physics_hybrid_master
COMMENT

python scripts/fast_batch_inference.py \
    --logs logs \
    --num_samples 5 \
    --steps 50 \
    --data_path data/uk_roof_dataset.h5 \
    --default_config configs/latent-diffusion/uk/cfd_ldm_uk.yaml \
    --outdir_suffix fast_inference_best_ckpt_test \
    --filter uk_grad_corr_low,uk_pinns_baseline,uk_physics_hybrid,uk_roof_balanced_hybrid_fixed,uk_pix2pix_sobel_hybrid
