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
    
    # Submit train_ldm.slurm which is now in scripts/ relative to root
    sbatch --export=ALL,EXTRA_ARGS="$FINAL_ARGS" scripts/train_ldm.slurm
}

# Experiment 1
submit_job "medlr_highb_medaux" "-n medlr_highb_medaux \
 model.base_learning_rate=2.0e-6 \
 data.params.batch_size=16 \
 model.params.original_elbo_weight=1.0e-4 \
 lightning.callbacks.image_logger.params.batch_frequency=2000 \
 lightning.trainer.log_every_n_steps=50"

# Experiment 2
submit_job "highlr_highb_medaux" "-n highlr_highb_medaux \
 model.base_learning_rate=5.0e-6 \
 data.params.batch_size=16 \
 model.params.original_elbo_weight=1.0e-4 \
 lightning.callbacks.image_logger.params.batch_frequency=2000 \
 lightning.trainer.log_every_n_steps=50"

# Experiment 3
submit_job "lowlr_highb_medaux" "-n lowlr_highb_medaux \
 model.base_learning_rate=1.0e-6 \
 data.params.batch_size=16 \
 model.params.original_elbo_weight=1.0e-4 \
 lightning.callbacks.image_logger.params.batch_frequency=2000 \
 lightning.trainer.log_every_n_steps=50"

# Experiment 4
submit_job "medlr_lowb_medaux" "-n medlr_lowb_medaux \
 model.base_learning_rate=2.0e-6 \
 data.params.batch_size=6 \
 model.params.original_elbo_weight=1.0e-4 \
 lightning.callbacks.image_logger.params.batch_frequency=2000 \
 lightning.trainer.log_every_n_steps=50"

# Experiment 5
submit_job "highlr_lowb_medaux" "-n highlr_lowb_medaux \
 model.base_learning_rate=5.0e-6 \
 data.params.batch_size=6 \
 model.params.original_elbo_weight=1.0e-4 \
 lightning.callbacks.image_logger.params.batch_frequency=2000 \
 lightning.trainer.log_every_n_steps=50"

# Experiment 6
submit_job "lowlr_lowb_medaux" "-n lowlr_lowb_medaux \
 model.base_learning_rate=1.0e-6 \
 data.params.batch_size=6 \
 model.params.original_elbo_weight=1.0e-3 \
 lightning.callbacks.image_logger.params.batch_frequency=2000 \
 lightning.trainer.log_every_n_steps=50"

# Experiment 7 (Note: Reused name 'medlr_lowb_medaux' in prompt, assuming user meant different weight or suffix. 
# Prompt had weight 1.0e-1. Renamed to medlr_lowb_highweight to avoid collision with Exp 4)
submit_job "medlr_lowb_highweight" "-n medlr_lowb_highweight \
 model.base_learning_rate=2.0e-6 \
 data.params.batch_size=6 \
 model.params.original_elbo_weight=1.0e-1 \
 lightning.callbacks.image_logger.params.batch_frequency=2000 \
 lightning.trainer.log_every_n_steps=50"

# Experiment 8
submit_job "medlr_lowb_highaux" "-n medlr_lowb_highaux \
 model.base_learning_rate=2.0e-6 \
 data.params.batch_size=6 \
 model.params.original_elbo_weight=1.0 \
 lightning.callbacks.image_logger.params.batch_frequency=2000 \
 lightning.trainer.log_every_n_steps=50"

# Experiment 9
submit_job "medlr_lowb_lowaux" "-n medlr_lowb_lowaux \
 model.base_learning_rate=2.0e-6 \
 data.params.batch_size=6 \
 model.params.original_elbo_weight=1.0e-4 \
 lightning.callbacks.image_logger.params.batch_frequency=2000 \
 lightning.trainer.log_every_n_steps=50"

# Experiment 10
submit_job "medlr_lowb_noaux" "-n medlr_lowb_noaux \
 model.base_learning_rate=2.0e-6 \
 data.params.batch_size=6 \
 model.params.original_elbo_weight=0 \
 lightning.callbacks.image_logger.params.batch_frequency=2000 \
 lightning.trainer.log_every_n_steps=50"
