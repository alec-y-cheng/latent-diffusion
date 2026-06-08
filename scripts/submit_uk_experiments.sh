#!/bin/bash

# Usage:
#   ./scripts/submit_uk_experiments.sh [--resume]
#   AE_CKPT=/path/to/autoencoder.ckpt ./scripts/submit_uk_experiments.sh [--resume]
# Submits all UK 4-channel LDM experiment variants

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR/.." || exit 1
echo "Running from project root: $(pwd)"

AE_CKPT=${AE_CKPT:-logs/2026-06-02T13-05-15_autoencoder_kl_32x32x4_uk/checkpoints/epoch=000062.ckpt}
if [ ! -f "$AE_CKPT" ]; then
    echo "Error: autoencoder checkpoint not found: $AE_CKPT"
    echo "Set AE_CKPT=/path/to/checkpoint.ckpt to choose a different first-stage checkpoint."
    exit 1
fi
AE_CKPT=$(realpath "$AE_CKPT")
echo "Using autoencoder checkpoint: $AE_CKPT"

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

get_latest_resume_target() {
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

strip_name_arg() {
    # main.py forbids using -n/--name together with -r/--resume.
    # The resumed logdir already carries the experiment name.
    echo "$1" | sed -E 's/(^|[[:space:]])(-n|--name)[[:space:]]+[^[:space:]]+//'
}

submit_job() {
    NAME=$1
    ARGS=$2
    CONFIG=$3
    FINAL_ARGS="-b $CONFIG $ARGS model.params.first_stage_config.params.ckpt_path=$AE_CKPT lightning.trainer.max_epochs=500"
    
    if [ "$RESUME_MODE" = true ]; then
        RESUME_TARGET=$(get_latest_resume_target "$NAME")
        if [ ! -z "$RESUME_TARGET" ]; then
            RESUME_ARGS=$(strip_name_arg "$ARGS")
            echo "Resuming $NAME in existing logdir from: $RESUME_TARGET"
            # Put -r immediately after the config so argparse stops -b/--base
            # from greedily consuming dotlist overrides as config paths.
            FINAL_ARGS="-b $CONFIG -r $RESUME_TARGET $RESUME_ARGS model.params.first_stage_config.params.ckpt_path=$AE_CKPT lightning.trainer.max_epochs=500"
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

#logs/2026-04-28T02-29-06_autoencoder_kl_32x32x4_uk/checkpoints/epoch=000097.ckpt

# --- Standard UNet (no physics) ---


submit_job "uk_roof_balanced_hybrid_fixed" "-n uk_roof_balanced_hybrid_fixed \
 model.params.use_pinn_loss=True \
 model.params.pinn_loss_weight=2.0 \
 model.params.grad_corr_weight=0.25 \
 model.params.grad_corr_per_channel=True \
 model.params.grad_corr_masked=True \
 model.params.pinn_sdf_channel=3 \
 model.params.pinn_building_channel=4 \
 model.params.pinn_building_mask_sharpness=100.0 \
 model.params.lambda_res=0.5 \
 model.params.lambda_bc=1.0 \
 model.params.lambda_smooth=0.10 \
 model.params.lambda_range=0.05 \
 model.params.lambda_masked_recon=1.0 \
 model.params.lambda_roof_background=0.25 \
 model.params.lambda_floor_background=0.10 \
 model.params.turbulence_smooth_weight=0.15 \
 model.params.roof_smooth_weight=0.25 \
 model.params.unet_config.params.use_wavelet=False \
 model.base_learning_rate=5.0e-7 \
 data.params.batch_size=16 \
 lightning.trainer.accumulate_grad_batches=4 \
 lightning.trainer.gradient_clip_val=1.0 \
 lightning.trainer.precision=16 \
 model.params.original_elbo_weight=5.0e-6 \
 lightning.callbacks.image_logger.params.batch_frequency=2000 \
 lightning.modelcheckpoint.params.save_top_k=3 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/uk/cfd_ldm_uk_pinnsformer.yaml"





submit_job "uk_pinns_baseline" "-n uk_pinns_baseline \
 model.params.use_pinn_loss=True \
 model.params.pinn_loss_weight=5.0 \
 model.params.grad_corr_weight=0.02 \
 model.params.lambda_res=1.0 \
 model.params.lambda_bc=1.0 \
 model.params.lambda_smooth=0.25 \
 model.par<<'COMMENT'ams.lambda_range=0.05 \
 model.params.unet_config.params.use_wavelet=False \
 model.base_learning_rate=5.0e-7 \
 data.params.batch_size=16 \
 lightning.trainer.accumulate_grad_batches=4 \
 lightning.trainer.gradient_clip_val=1.0 \
 lightning.trainer.precision=16 \
 model.params.original_elbo_weight=5.0e-6 \
 lightning.callbacks.image_logger.params.batch_frequency=2000 \
 lightning.modelcheckpoint.params.save_top_k=3 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/uk/cfd_ldm_uk_pinnsformer.yaml"


submit_job "uk_grad_corr_low" "-n uk_grad_corr_low \
 model.params.grad_corr_weight=0.02 \
 model.base_learning_rate=5.0e-7 \
 data.params.batch_size=16 \
 lightning.trainer.accumulate_grad_batches=4 \
 lightning.trainer.gradient_clip_val=1.0 \
 model.params.original_elbo_weight=5.0e-6 \
 lightning.trainer.precision=16 \
 lightning.callbacks.image_logger.params.batch_frequency=2000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/uk/cfd_ldm_uk.yaml"


submit_job "uk_physics_hybrid" "-n uk_physics_hybrid \
 model.params.use_pinn_loss=True \
 model.params.pinn_loss_weight=5.0 \
 model.params.grad_corr_weight=0.5 \
 model.base_learning_rate=5.0e-7 \
 data.params.batch_size=16 \
 lightning.trainer.accumulate_grad_batches=4 \
 lightning.trainer.gradient_clip_val=1.0 \
 lightning.trainer.precision=16 \
 model.params.original_elbo_weight=5.0e-6 \
 lightning.callbacks.image_logger.params.batch_frequency=2000 \
 lightning.modelcheckpoint.params.save_top_k=3 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/uk/cfd_ldm_uk_pinnsformer.yaml"


 <<'COMMENT'

submit_job "uk_grad_corr_med" "-n uk_grad_corr_med \
 model.params.grad_corr_weight=0.5 \
 model.base_learning_rate=5.0e-7 \
 data.params.batch_size=8 \
 lightning.trainer.accumulate_grad_batches=8 \
 lightning.trainer.gradient_clip_val=1.0 \
 model.params.original_elbo_weight=5.0e-6 \
 lightning.callbacks.image_logger.params.batch_frequency=625 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/uk/cfd_ldm_uk.yaml"

submit_job "uk_grad_corr_high" "-n uk_grad_corr_high \
 model.params.grad_corr_weight=1.0 \
 model.base_learning_rate=5.0e-7 \
 data.params.batch_size=8 \
 lightning.trainer.accumulate_grad_batches=8 \
 lightning.trainer.gradient_clip_val=1.0 \
 model.params.original_elbo_weight=5.0e-6 \
 lightning.callbacks.image_logger.params.batch_frequency=625 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/uk/cfd_ldm_uk.yaml"





COMMENT

# --- PINNsformer (physics-informed, wavelet UNet) ---


# --- DID variants (uncomment when DID UK dataset is ready) ---
# submit_job "uk_grad_corr_med_did" "-n uk_grad_corr_med_did \
#  model.params.grad_corr_weight=0.5 \
#  model.base_learning_rate=5.0e-7 \
#  data.params.batch_size=8 \
#  lightning.trainer.accumulate_grad_batches=8 \
#  lightning.trainer.gradient_clip_val=1.0 \
#  model.params.original_elbo_weight=5.0e-6 \
#  lightning.callbacks.image_logger.params.batch_frequency=625 \
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
#  lightning.callbacks.image_logger.params.batch_frequency=625 \
#  lightning.modelcheckpoint.params.save_top_k=1 \
#  lightning.trainer.log_every_n_steps=50" "configs/latent-diffusion/uk/cfd_ldm_uk_pinnsformer_did.yaml"
