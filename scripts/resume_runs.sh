#!/bin/bash

# Usage: ./scripts/submit_gradcorr_runs.sh [--resume]

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
    
    # Iterate through all log dirs for this experiment, newest first
    for LATEST_LOGDIR in $(ls -td "${LOGS_DIR}"/*${EXP_NAME}* 2>/dev/null); do
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
    done
    
    return 1
}

submit_job() {
    NAME=$1
    SLURM_SCRIPT=$2
    ARGS=$3
    
    FINAL_ARGS="$ARGS"
    
    if [ "$RESUME_MODE" = true ]; then
        CKPT=$(get_latest_ckpt "$NAME")
        if [ ! -z "$CKPT" ]; then
            echo "Resuming $NAME from $CKPT"
            FINAL_ARGS="$FINAL_ARGS --resume_from_checkpoint $CKPT"
        else
            echo "Warning: No checkpoint found for $NAME. Starting fresh."
        fi
    fi
    
    # Submit the specific slurm script
    sbatch --export=ALL,EXTRA_ARGS="$FINAL_ARGS" "$SLURM_SCRIPT"
}

# --- Experiment: Wavelet ---
submit_job "cfd_ldm_pinnsformer" "scripts/train_ldm_pinnsformer.slurm" "-n cfd_ldm_pinnsformer \
 model.base_learning_rate=5.0e-6 \
 data.params.batch_size=16 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50"

 submit_job "low_grad_corr" "scripts/train_ldm.slurm" "-n low_grad_corr \
 model.params.grad_corr_weight=0.1 \
 model.base_learning_rate=2.0e-6 \
 data.params.batch_size=16 \
 model.params.original_elbo_weight=1.0e-4 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50"

# --- Experiment 2: Med GradCorr ---
submit_job "med_grad_corr" "scripts/train_ldm.slurm" "-n med_grad_corr \
 model.params.grad_corr_weight=0.5 \
 model.base_learning_rate=2.0e-6 \
 data.params.batch_size=16 \
 model.params.original_elbo_weight=1.0e-4 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50"

# --- Experiment 3: High GradCorr ---
submit_job "high_grad_corr" "scripts/train_ldm.slurm" "-n high_grad_corr \
 model.params.grad_corr_weight=1 \
 model.base_learning_rate=2.0e-6 \
 data.params.batch_size=16 \
 model.params.original_elbo_weight=1.0e-4 \
 lightning.callbacks.image_logger.params.batch_frequency=10000 \
 lightning.modelcheckpoint.params.save_top_k=1 \
 lightning.trainer.log_every_n_steps=50"