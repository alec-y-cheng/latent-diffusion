#!/bin/bash

# Usage: ./scripts/submit_all_experiments.sh [--resume]

# Ensure we are in the project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.." || exit 1
echo "Running from project root: $(pwd)"

RESUME_MODE=false
if [ "$1" == "--resume" ]; then
    RESUME_MODE=true
    echo "Resume mode enabled. Searching for latest checkpoints..."
fi

get_latest_ckpt() {
    EXP_NAME=$1
    # Check logs directory in current (root) path
    LOGS_DIR="logs"
    
    # Find latest log dir for this experiment
    LATEST_LOGDIR=$(ls -td "${LOGS_DIR}"/*${EXP_NAME}* 2>/dev/null | head -n 1)
    if [ -z "$LATEST_LOGDIR" ]; then
        return 1
    fi
    
    # Check for last.ckpt first, then any ckpt
    CKPT="${LATEST_LOGDIR}/checkpoints/last.ckpt"
    if [ -f "$CKPT" ]; then
        echo "$CKPT"
        return 0
    fi
    
    # Fallback to any ckpt
    CKPT=$(ls -t ${LATEST_LOGDIR}/checkpoints/*.ckpt 2>/dev/null | head -n 1)
    if [ ! -z "$CKPT" ]; then
        echo "$CKPT"
        return 0
    fi
    
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

 submit_job "grad_corr_low_real" "-n grad_corr_low \
 model.params.grad_corr_weight=0.1 \
 model.base_learning_rate=1.0e-5 \
 data.params.batch_size=64 \
 model.params.original_elbo_weight=1.0e-4 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/cfd_ldm.yaml"

submit_job "grad_corr_med_real" "-n grad_corr_med \
 model.params.grad_corr_weight=0.5 \
 model.base_learning_rate=1.0e-5 \
 data.params.batch_size=64 \
 model.params.original_elbo_weight=1.0e-4 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/cfd_ldm.yaml"

submit_job "grad_corr_high_real" "-n grad_corr_high \
 model.params.grad_corr_weight=1 \
 model.base_learning_rate=1.0e-5 \
 data.params.batch_size=64 \
 model.params.original_elbo_weight=1.0e-4 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/cfd_ldm.yaml"

# 1. PINNs Baseline (Testing the newly wired physics loss WITHOUT wavelet confounding it)
submit_job "pinns_baseline" "-n pinns_baseline \
 model.params.use_pinn_loss=True \
 model.params.pinn_loss_weight=0.1 \
 model.params.lambda_res=1.0 \
 model.params.lambda_bc=1.0 \
 model.params.unet_config.params.use_wavelet=False \
 model.base_learning_rate=1.0e-5 \
 data.params.batch_size=64 \
 model.params.original_elbo_weight=1.0e-4 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/cfd_ldm_pinnsformer.yaml"

# 2. PINNs Boundary Condition Focus (Strictly respecting physics, no wavelet)
submit_job "pinns_bc_heavy" "-n pinns_bc_heavy \
 model.params.use_pinn_loss=True \
 model.params.pinn_loss_weight=0.5 \
 model.params.lambda_res=1.0 \
 model.params.lambda_bc=5.0 \
 model.params.unet_config.params.use_wavelet=False \
 model.base_learning_rate=1.0e-5 \
 data.params.batch_size=64 \
 model.params.original_elbo_weight=1.0e-4 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/cfd_ldm_pinnsformer.yaml"

# 3. Wavelet + Grad Corr Combo (Tests if frequency-domain processing synergizes with physical gradient tracking)
submit_job "wavelet_grad_corr" "-n wavelet_grad_corr \
 model.params.grad_corr_weight=0.5 \
 model.params.use_pinn_loss=False \
 model.base_learning_rate=1.0e-5 \
 data.params.batch_size=64 \
 model.params.original_elbo_weight=1.0e-4 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/cfd_ldm_wavelet.yaml"

# 4. Ultimate Physics Combo (PINNs + Grad Corr + Wavelet Dataset)
submit_job "physics_hybrid_master" "-n physics_hybrid_master \
 model.params.use_pinn_loss=True \
 model.params.pinn_loss_weight=0.5 \
 model.params.grad_corr_weight=0.5 \
 model.base_learning_rate=1.0e-5 \
 data.params.batch_size=64 \
 model.params.original_elbo_weight=1.0e-4 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/cfd_ldm_pinnsformer.yaml"