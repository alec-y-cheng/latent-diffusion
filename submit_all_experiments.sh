#!/bin/bash

# Experiment 1
sbatch --export=ALL,EXTRA_ARGS="-n medlr_highb_medaux \
 --model.base_learning_rate=2.0e-6 \
 --data.params.batch_size=16 \
 --model.params.original_elbo_weight=1.0e-4 \
 --lightning.callbacks.image_logger.params.batch_frequency=2000 \
 --lightning.trainer.log_every_n_steps=50" train_ldm.slurm

# Experiment 2
sbatch --export=ALL,EXTRA_ARGS="-n highlr_highb_medaux \
 --model.base_learning_rate=5.0e-6 \
 --data.params.batch_size=16 \
 --model.params.original_elbo_weight=1.0e-4 \
 --lightning.callbacks.image_logger.params.batch_frequency=2000 \
 --lightning.trainer.log_every_n_steps=50" train_ldm.slurm

# Experiment 3
sbatch --export=ALL,EXTRA_ARGS="-n lowlr_highb_medaux \
 --model.base_learning_rate=1.0e-6 \
 --data.params.batch_size=16 \
 --model.params.original_elbo_weight=1.0e-4 \
 --lightning.callbacks.image_logger.params.batch_frequency=2000 \
 --lightning.trainer.log_every_n_steps=50" train_ldm.slurm

# Experiment 4
sbatch --export=ALL,EXTRA_ARGS="-n medlr_lowb_medaux \
 --model.base_learning_rate=2.0e-6 \
 --data.params.batch_size=6 \
 --model.params.original_elbo_weight=1.0e-4 \
 --lightning.callbacks.image_logger.params.batch_frequency=2000 \
 --lightning.trainer.log_every_n_steps=50" train_ldm.slurm

# Experiment 5
sbatch --export=ALL,EXTRA_ARGS="-n highlr_lowb_medaux \
 --model.base_learning_rate=5.0e-6 \
 --data.params.batch_size=6 \
 --model.params.original_elbo_weight=1.0e-4 \
 --lightning.callbacks.image_logger.params.batch_frequency=2000 \
 --lightning.trainer.log_every_n_steps=50" train_ldm.slurm

# Experiment 6
sbatch --export=ALL,EXTRA_ARGS="-n lowlr_lowb_medaux \
 --model.base_learning_rate=1.0e-6 \
 --data.params.batch_size=6 \
 --model.params.original_elbo_weight=1.0e-3 \
 --lightning.callbacks.image_logger.params.batch_frequency=2000 \
 --lightning.trainer.log_every_n_steps=50" train_ldm.slurm

# Experiment 7 (Note: Reused name 'medlr_lowb_medaux' in prompt, assuming user meant different weight or suffix. 
# Prompt had weight 1.0e-1. Renamed to medlr_lowb_highweight to avoid collision with Exp 4)
sbatch --export=ALL,EXTRA_ARGS="-n medlr_lowb_highweight \
 --model.base_learning_rate=2.0e-6 \
 --data.params.batch_size=6 \
 --model.params.original_elbo_weight=1.0e-1 \
 --lightning.callbacks.image_logger.params.batch_frequency=2000 \
 --lightning.trainer.log_every_n_steps=50" train_ldm.slurm

# Experiment 8
sbatch --export=ALL,EXTRA_ARGS="-n medlr_lowb_highaux \
 --model.base_learning_rate=2.0e-6 \
 --data.params.batch_size=6 \
 --model.params.original_elbo_weight=1.0 \
 --lightning.callbacks.image_logger.params.batch_frequency=2000 \
 --lightning.trainer.log_every_n_steps=50" train_ldm.slurm

# Experiment 9
sbatch --export=ALL,EXTRA_ARGS="-n medlr_lowb_lowaux \
 --model.base_learning_rate=2.0e-6 \
 --data.params.batch_size=6 \
 --model.params.original_elbo_weight=1.0e-4 \
 --lightning.callbacks.image_logger.params.batch_frequency=2000 \
 --lightning.trainer.log_every_n_steps=50" train_ldm.slurm

# Experiment 10
sbatch --export=ALL,EXTRA_ARGS="-n medlr_lowb_noaux \
 --model.base_learning_rate=2.0e-6 \
 --data.params.batch_size=6 \
 --model.params.original_elbo_weight=0 \
 --lightning.callbacks.image_logger.params.batch_frequency=2000 \
 --lightning.trainer.log_every_n_steps=50" train_ldm.slurm
