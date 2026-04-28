#!/bin/bash
#SBATCH --job-name=uk_npz_to_h5
#SBATCH --partition=coe-gpu
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/convert_uk_npz_to_h5_%j.out

module load cuda/11.8
source $HOME/.bashrc
conda activate ldm

export PYTHONUNBUFFERED=1

python3 -c "
import numpy as np
import h5py
import time
import os
import zipfile

npz_path = 'data/uk_set/expanded_dataset_uk_roof.npz'
h5_path  = 'data/uk_set/dataset_uk.h5'

print(f'Starting conversion: {npz_path} -> {h5_path}')
t_start = time.time()

# We use zipfile directly to read X and Y from the NPZ to avoid OOM by mmap_mode='r'
# which sometimes still triggers aggressive RAM usage on large NPZs.
with zipfile.ZipFile(npz_path, 'r') as zf:
    # 1. Get Shapes and Dtypes
    import numpy.lib.format
    with zf.open('X.npy', 'r') as f:
        version = numpy.lib.format.read_magic(f)
        x_shape, _, x_dtype = numpy.lib.format._read_array_header(f, version)
    with zf.open('Y.npy', 'r') as f:
        version = numpy.lib.format.read_magic(f)
        y_shape, _, y_dtype = numpy.lib.format._read_array_header(f, version)
    
    print(f'  X Detected: {x_shape}, {x_dtype}')
    print(f'  Y Detected: {y_shape}, {y_dtype}')
    
    N = x_shape[0]
    CHUNK = 100
    
    with h5py.File(h5_path, 'w') as f_h5:
        ds_x = f_h5.create_dataset('X', shape=x_shape, dtype=x_dtype,
                                chunks=(1, *x_shape[1:]),
                                compression='gzip', compression_opts=1)
        ds_y = f_h5.create_dataset('Y', shape=y_shape, dtype=y_dtype,
                                chunks=(1, *y_shape[1:]),
                                compression='gzip', compression_opts=1)
        
        # We reopen X and Y effectively to stream them
        # Note: This is efficient for ZipFiles if we read sequentially
        with zf.open('X.npy', 'r') as fx, zf.open('Y.npy', 'r') as fy:
            # Skip headers
            numpy.lib.format.read_magic(fx)
            numpy.lib.format._read_array_header(fx, version) # Dummy read
            numpy.lib.format.read_magic(fy)
            numpy.lib.format._read_array_header(fy, version) # Dummy read
            
            x_step = np.prod(x_shape[1:]) * 4
            y_step = np.prod(y_shape[1:]) * 4
            
            for start in range(0, N, CHUNK):
                end = min(start + CHUNK, N)
                current_count = end - start
                
                # Read binary chunks and convert to numpy
                bx = fx.read(x_step * current_count)
                by = fy.read(y_step * current_count)
                
                nx = np.frombuffer(bx, dtype=x_dtype).reshape((current_count, *x_shape[1:]))
                ny = np.frombuffer(by, dtype=y_dtype).reshape((current_count, *y_shape[1:]))
                
                ds_x[start:end] = nx
                ds_y[start:end] = ny
                
                elapsed = time.time() - t_start
                print(f'  [{end/N*100:5.1f}%] Wrote samples {start}-{end-1} ({elapsed:.0f}s elapsed)')

print(f'Conversion Complete. Total time: {time.time()-t_start:.1f}s')
print(f'H5 File size: {os.path.getsize(h5_path)/1e9:.2f} GB')
"
