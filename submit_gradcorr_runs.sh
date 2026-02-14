#!/bin/bash
# Experiment 1
sbatch --export=ALL,EXTRA_ARGS="-n low_grad_corr \
 model.params.grad_corr_weight=0.1 \
 model.base_learning_rate=2.0e-6 \
 data.params.batch_size=16 \
 model.params.original_elbo_weight=1.0e-4 \
 lightning.callbacks.image_logger.params.batch_frequency=2000 \
 lightning.trainer.log_every_n_steps=50" train_ldm.slurm

# Experiment 1
sbatch --export=ALL,EXTRA_ARGS="-n med_grad_corr \
 model.params.grad_corr_weight=0.5 \
 model.base_learning_rate=2.0e-6 \
 data.params.batch_size=16 \
 model.params.original_elbo_weight=1.0e-4 \
 lightning.callbacks.image_logger.params.batch_frequency=2000 \
 lightning.trainer.log_every_n_steps=50" train_ldm.slurm

# Experiment 1
sbatch --export=ALL,EXTRA_ARGS="-n high_grad_corr \
 model.params.grad_corr_weight=1 \
 model.base_learning_rate=2.0e-6 \
 data.params.batch_size=16 \
 model.params.original_elbo_weight=1.0e-4 \
 lightning.callbacks.image_logger.params.batch_frequency=2000 \
 lightning.trainer.log_every_n_steps=50" train_ldm.slurm