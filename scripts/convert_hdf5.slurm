#!/bin/bash
#SBATCH --job-name=convert_h5
##SBATCH --partition=coe-gpu
##SBATCH --qos=coe-ice
##SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/convert_h5_%j.out

#SBATCH --account=gts-pkastner3
#SBATCH --qos=embers
#SBATCH --partition=gpu-h200
#SBATCH --gres=gpu:h200:1

module load cuda/11.8
source $HOME/.bashrc
conda activate ldm

echo "Starting HDF5 conversion from chunks..."
python convert_to_hdf5.py
echo "Finished!"
