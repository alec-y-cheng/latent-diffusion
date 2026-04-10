#!/bin/bash

# Usage: ./scripts/submit_all_experiments.sh [--resume]

# Ensure we are in the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.." || exit 1
echo "Running from project root: $(pwd)"

RESUME_MODE=false
for arg in "$@"; do
    if [ "$arg" == "--resume" ]; then
        RESUME_MODE=true
        echo "Resume mode enabled. Searching for latest checkpoints..."
        break
    fi
done

get_latest_ckpt() {
    EXP_NAME=$1
    LOGS_DIR="logs"
    
    # Find all log dirs for this experiment, sorted by time (newest first)
    # Filter for directories only to avoid matching log files
    ALL_LOGDIRS=$(ls -td "${LOGS_DIR}"/*"${EXP_NAME}"* 2>/dev/null)
    
    for LATEST_LOGDIR in $ALL_LOGDIRS; do
        if [ ! -d "$LATEST_LOGDIR" ]; then continue; fi
        
        # Check for last.ckpt first
        CKPT="${LATEST_LOGDIR}/checkpoints/last.ckpt"
        if [ -f "$CKPT" ]; then
            realpath "$CKPT"
            return 0
        fi
        
        # Fallback to the most recent epoch checkpoint
        CKPT=$(ls -t "${LATEST_LOGDIR}"/checkpoints/*.ckpt 2>/dev/null | head -n 1)
        if [ -n "$CKPT" ] && [ -f "$CKPT" ]; then
            realpath "$CKPT"
            return 0
        fi
    done
    
    return 1
}

submit_job() {
    NAME=$1
    ARGS=$2
    CONFIG=${3:-"configs/latent-diffusion/cfd_ldm.yaml"}
    
    FINAL_ARGS="-b $CONFIG $ARGS"
    
    if [ "$RESUME_MODE" = true ]; then
        CKPT=$(get_latest_ckpt "$NAME")
        if [ ! -z "$CKPT" ]; then
            echo "Resuming $NAME from $CKPT"
            FINAL_ARGS="$FINAL_ARGS --resume_from_checkpoint $CKPT"
        else
            echo "Warning: No checkpoint found for $NAME. Starting fresh."
        fi
    fi
    
    # Submit train_ldm.slurm which is now in scripts/ relative to root
    sbatch --export=ALL,EXTRA_ARGS="$FINAL_ARGS" scripts/train_ldm.slurm
}


# LR note: best vanilla was 5e-6 with bs=6, accum=1 → effective=3e-5
# New setup: bs=8, accum=8 → base_lr = 3e-5 / 64 = 4.69e-7 → 5e-7
# gradient_clip_val=1.0 added to prevent NaN from physics loss gradients


# PINNs runs use precision=32 — Laplacian second-order diff is numerically fragile in fp16

submit_job "32b_5e6lr" "-n 32b_5e6lr \
 model.params.grad_corr_weight=0.5 \
 model.base_learning_rate=5.0e-7 \
 data.params.batch_size=32 \
 lightning.trainer.gradient_clip_val=1.0 \
 model.params.original_elbo_weight=1.0e-4 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/cfd_ldm.yaml"


submit_job "grad_corr_low_highlr" "-n grad_corr_low_highlr \
 model.params.grad_corr_weight=0.1 \
 model.base_learning_rate=1.0e-6 \
 data.params.batch_size=8 \
 lightning.trainer.accumulate_grad_batches=8 \
 lightning.trainer.gradient_clip_val=1.0 \
 model.params.original_elbo_weight=5.0e-6 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/cfd_ldm.yaml"


submit_job "grad_corr_low_noclip" "-n grad_corr_low_noclip \
 model.params.grad_corr_weight=0.1 \
 model.base_learning_rate=5.0e-7 \
 data.params.batch_size=8 \
 lightning.trainer.accumulate_grad_batches=8 \
 lightning.trainer.gradient_clip_val=0.0 \
 model.params.original_elbo_weight=5.0e-6 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/cfd_ldm.yaml"

submit_job "pinns_baseline_base" "-n pinns_baseline_base \
 model.params.use_pinn_loss=True \
 model.params.pinn_loss_weight=0.1 \
 model.params.lambda_res=1.0 \
 model.params.lambda_bc=1.0 \
 model.params.unet_config.params.use_wavelet=False \
 model.base_learning_rate=5.0e-7 \
 data.params.batch_size=8 \
 lightning.trainer.accumulate_grad_batches=8 \
 lightning.trainer.gradient_clip_val=1.0 \
 lightning.trainer.precision=32 \
 model.params.original_elbo_weight=5.0e-6 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/cfd_ldm_pinnsformer.yaml"


 submit_job "grad_corr_low_real" "-n grad_corr_low_real \
 model.params.grad_corr_weight=0.1 \
 model.base_learning_rate=5.0e-7 \
 data.params.batch_size=8 \
 lightning.trainer.accumulate_grad_batches=8 \
 lightning.trainer.gradient_clip_val=1.0 \
 model.params.original_elbo_weight=5.0e-6 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/cfd_ldm.yaml"

submit_job "grad_corr_med_real" "-n grad_corr_med_real \
 model.params.grad_corr_weight=0.5 \
 model.base_learning_rate=5.0e-7 \
 data.params.batch_size=8 \
 lightning.trainer.accumulate_grad_batches=8 \
 lightning.trainer.gradient_clip_val=1.0 \
 model.params.original_elbo_weight=5.0e-6 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/cfd_ldm.yaml"

submit_job "grad_corr_high_real" "-n grad_corr_high_real \
 model.params.grad_corr_weight=1 \
 model.base_learning_rate=5.0e-7 \
 data.params.batch_size=8 \
 lightning.trainer.accumulate_grad_batches=8 \
 lightning.trainer.gradient_clip_val=1.0 \
 model.params.original_elbo_weight=5.0e-6 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/cfd_ldm.yaml"


submit_job "pinns_baseline_wavelet" "-n pinns_baseline_wavelet \
 model.params.use_pinn_loss=True \
 model.params.pinn_loss_weight=0.1 \
 model.params.lambda_res=1.0 \
 model.params.lambda_bc=1.0 \
 model.params.unet_config.params.use_wavelet=True \
 model.base_learning_rate=5.0e-7 \
 data.params.batch_size=8 \
 lightning.trainer.accumulate_grad_batches=8 \
 lightning.trainer.gradient_clip_val=1.0 \
 lightning.trainer.precision=32 \
 model.params.original_elbo_weight=5.0e-6 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/cfd_ldm_pinnsformer.yaml"

submit_job "pinns_bc_heavy" "-n pinns_bc_heavy \
 model.params.use_pinn_loss=True \
 model.params.pinn_loss_weight=0.5 \
 model.params.lambda_res=1.0 \
 model.params.lambda_bc=5.0 \
 model.params.unet_config.params.use_wavelet=False \
 model.base_learning_rate=5.0e-7 \
 data.params.batch_size=8 \
 lightning.trainer.accumulate_grad_batches=8 \
 lightning.trainer.gradient_clip_val=1.0 \
 lightning.trainer.precision=32 \
 model.params.original_elbo_weight=5.0e-6 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/cfd_ldm_pinnsformer.yaml"

submit_job "wavelet_grad_corr" "-n wavelet_grad_corr \
 model.params.grad_corr_weight=0.5 \
 model.params.use_pinn_loss=False \
 model.base_learning_rate=5.0e-7 \
 data.params.batch_size=8 \
 lightning.trainer.accumulate_grad_batches=8 \
 lightning.trainer.gradient_clip_val=1.0 \
 model.params.original_elbo_weight=5.0e-6 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/cfd_ldm_wavelet.yaml"

submit_job "physics_hybrid_master" "-n physics_hybrid_master \
 model.params.use_pinn_loss=True \
 model.params.pinn_loss_weight=0.5 \
 model.params.grad_corr_weight=0.5 \
 model.base_learning_rate=5.0e-7 \
 data.params.batch_size=8 \
 lightning.trainer.accumulate_grad_batches=8 \
 lightning.trainer.gradient_clip_val=1.0 \
 lightning.trainer.precision=32 \
 model.params.original_elbo_weight=5.0e-6 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/cfd_ldm_pinnsformer.yaml"

<<'COMMENT'
COMMENT