#!/bin/bash

# Usage: ./scripts/submit_uk_experiments.sh [--resume]
# Submits all UK 4-channel LDM experiment variants

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
    
    ALL_LOGDIRS=$(ls -td "${LOGS_DIR}"/*_"${EXP_NAME}" 2>/dev/null)
    
    for LATEST_LOGDIR in $ALL_LOGDIRS; do
        if [ ! -d "$LATEST_LOGDIR" ]; then continue; fi
        
        CKPT="${LATEST_LOGDIR}/checkpoints/last.ckpt"
        if [ -f "$CKPT" ]; then
            realpath "$CKPT"
            return 0
        fi
        
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
    CONFIG=$3
    FINAL_ARGS="-b $CONFIG $ARGS lightning.trainer.max_epochs=500"
    
    if [ "$RESUME_MODE" = true ]; then
        CKPT=$(get_latest_ckpt "$NAME")
        if [ ! -z "$CKPT" ]; then
            echo "Resuming $NAME from $CKPT"
            FINAL_ARGS="$FINAL_ARGS --resume_from_checkpoint $CKPT"
        else
            echo "Warning: No checkpoint found for $NAME. Starting fresh."
        fi
    fi
    
    echo "Submitting: $NAME"
    sbatch --export=ALL,EXTRA_ARGS="$FINAL_ARGS" scripts/train_ldm_uk.slurm
}

# ============================================================================
# UK LDM Experiments (4-channel targets: floor speed, floor turbulence, 
#                                         roof speed, roof turbulence)
# ============================================================================

# --- Standard UNet (no physics) ---

submit_job "uk_grad_corr_low" "-n uk_grad_corr_low \
 model.params.grad_corr_weight=0.1 \
 model.base_learning_rate=5.0e-7 \
 data.params.batch_size=8 \
 lightning.trainer.accumulate_grad_batches=8 \
 lightning.trainer.gradient_clip_val=1.0 \
 model.params.original_elbo_weight=5.0e-6 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/uk/cfd_ldm_uk.yaml"
 <<'COMMENT'
submit_job "uk_grad_corr_med" "-n uk_grad_corr_med \
 model.params.grad_corr_weight=0.5 \
 model.base_learning_rate=5.0e-7 \
 data.params.batch_size=8 \
 lightning.trainer.accumulate_grad_batches=8 \
 lightning.trainer.gradient_clip_val=1.0 \
 model.params.original_elbo_weight=5.0e-6 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/uk/cfd_ldm_uk.yaml"

submit_job "uk_grad_corr_high" "-n uk_grad_corr_high \
 model.params.grad_corr_weight=1.0 \
 model.base_learning_rate=5.0e-7 \
 data.params.batch_size=8 \
 lightning.trainer.accumulate_grad_batches=8 \
 lightning.trainer.gradient_clip_val=1.0 \
 model.params.original_elbo_weight=5.0e-6 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/uk/cfd_ldm_uk.yaml"
COMMENT

# --- PINNsformer (physics-informed, wavelet UNet) ---

submit_job "uk_pinns_baseline" "-n uk_pinns_baseline \
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
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/uk/cfd_ldm_uk_pinnsformer.yaml"


 <<'COMMENT'


submit_job "uk_pinns_bc_heavy" "-n uk_pinns_bc_heavy \
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
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/uk/cfd_ldm_uk_pinnsformer.yaml"

submit_job "uk_physics_hybrid" "-n uk_physics_hybrid \
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
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/uk/cfd_ldm_uk_pinnsformer.yaml"
COMMENT

# --- DID variants (uncomment when DID UK dataset is ready) ---
# submit_job "uk_grad_corr_med_did" "-n uk_grad_corr_med_did \
#  model.params.grad_corr_weight=0.5 \
#  model.base_learning_rate=5.0e-7 \
#  data.params.batch_size=8 \
#  lightning.trainer.accumulate_grad_batches=8 \
#  lightning.trainer.gradient_clip_val=1.0 \
#  model.params.original_elbo_weight=5.0e-6 \
#  lightning.callbacks.image_logger.params.batch_frequency=10000 \
#  lightning.modelcheckpoint.params.save_top_k=1 \
#  lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/uk/cfd_ldm_uk_did.yaml"

# submit_job "uk_physics_hybrid_did" "-n uk_physics_hybrid_did \
#  model.params.use_pinn_loss=True \
#  model.params.pinn_loss_weight=0.5 \
#  model.params.grad_corr_weight=0.5 \
#  model.base_learning_rate=5.0e-7 \
#  data.params.batch_size=8 \
#  lightning.trainer.accumulate_grad_batches=8 \
#  lightning.trainer.gradient_clip_val=1.0 \
#  lightning.trainer.precision=32 \
#  model.params.original_elbo_weight=5.0e-6 \
#  lightning.callbacks.image_logger.params.batch_frequency=10000 \
#  lightning.modelcheckpoint.params.save_top_k=1 \
#  lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/uk/cfd_ldm_uk_pinnsformer_did.yaml"
