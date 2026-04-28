#!/bin/bash
#SBATCH --job-name=ldm_uk
#SBATCH --partition=coe-gpu
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=012:00:00
#SBATCH --output=logs/train_%j.out

module load cuda/11.8
source $HOME/.bashrc
conda activate ldm

export CUDA_VISIBLE_DEVICES=0

# UK dataset is a single HDF5 file — copy it to local NVMe for fast IO
DATA_FILE=${DATA_FILE:-data/uk_set/dataset_uk.h5}

echo "Copying UK dataset to local NVMe ($TMPDIR)..."
mkdir -p $TMPDIR/$(dirname $DATA_FILE)
cp $DATA_FILE $TMPDIR/$DATA_FILE
echo "Copy complete. Size: $(du -h $TMPDIR/$DATA_FILE | cut -f1)"

python main.py \
  -t \
  --gpus 0, \
  $EXTRA_ARGS \
  data.params.train.params.data_path=$TMPDIR/$DATA_FILE \
  data.params.validation.params.data_path=$TMPDIR/$DATA_FILE
