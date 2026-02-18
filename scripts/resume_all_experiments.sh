#!/bin/bash

# Function to find the latest checkpoint for a given experiment name suffix
resume_experiment() {
    EXP_NAME=$1
    # Search for directories containing the experiment name in logs/
    # Sort by time (newest first) and pick the first one
    LATEST_LOGDIR=$(ls -td logs/*${EXP_NAME}* 2>/dev/null | head -n 1)

    if [ -z "$LATEST_LOGDIR" ]; then
        echo "Warning: No log directory found for experiment '$EXP_NAME'. Skipping."
        return
    fi

    CKPT_PATH="${LATEST_LOGDIR}/checkpoints/last.ckpt"
    
    if [ ! -f "$CKPT_PATH" ]; then
        echo "Warning: Checkpoint 'last.ckpt' not found in '$LATEST_LOGDIR'. Checking for other .ckpt files..."
        # Try to find any other .ckpt file, sorting by modification time to get latest
        CKPT_PATH=$(ls -t ${LATEST_LOGDIR}/checkpoints/*.ckpt 2>/dev/null | head -n 1)
        if [ -z "$CKPT_PATH" ]; then
             echo "Error: No checkpoints found in '$LATEST_LOGDIR'. Cannot resume."
             return
        fi
    fi

    echo "Resuming '$EXP_NAME' from: $CKPT_PATH"
    # Submit the job with RESUME variable
    sbatch --export=ALL,RESUME=$CKPT_PATH train_ldm.slurm
}

# List of experiment names (must match the -n names in submit_all_experiments.sh)
EXPERIMENTS=(
    "medlr_highb_medaux"
    "highlr_highb_medaux"
    "lowlr_highb_medaux"
    "medlr_lowb_medaux"
    "highlr_lowb_medaux"
    "lowlr_lowb_medaux"
    "medlr_lowb_highweight"
    "medlr_lowb_highaux"
    "medlr_lowb_lowaux"
    "medlr_lowb_noaux"
)

# Loop through all experiments and attempt to resume
for EXP in "${EXPERIMENTS[@]}"; do
    resume_experiment "$EXP"
done
