#!/bin/bash
#SBATCH --job-name=npz_to_h5
#SBATCH --partition=coe-gpu
#SBATCH --qos=coe-ice
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/convert_npz_to_h5_%j.out

module load cuda/11.8
source $HOME/.bashrc
conda activate ldm

export PYTHONUNBUFFERED=1

python3 -c "
import numpy as np
import h5py
import time
import os

npz_path = 'data/augmented/full_dataset.npz'
h5_path  = 'data/augmented/dataset.h5'

print('Memory-mapping NPZ file...')
t0 = time.time()
data = np.load(npz_path, mmap_mode='r')
X = data['X']
Y = data['Y']
print(f'  Mapped in {time.time()-t0:.1f}s')
print(f'  X: shape={X.shape}, dtype={X.dtype}')
print(f'  Y: shape={Y.shape}, dtype={Y.dtype}')

N = X.shape[0]
CHUNK = 100

print(f'Creating COMPRESSED HDF5 file (gzip-1), streaming {N} samples in chunks of {CHUNK}...')
t0 = time.time()
with h5py.File(h5_path, 'w') as f:
    # gzip level 1: fast compression, ~50-70% size reduction vs uncompressed
    # Still much faster than NPZ for random access due to per-chunk decompression
    ds_x = f.create_dataset('X', shape=X.shape, dtype=X.dtype,
                            chunks=(1, *X.shape[1:]),
                            compression='gzip', compression_opts=1)
    ds_y = f.create_dataset('Y', shape=Y.shape, dtype=Y.dtype,
                            chunks=(1, *Y.shape[1:]),
                            compression='gzip', compression_opts=1)
    
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        ds_x[start:end] = np.array(X[start:end])
        ds_y[start:end] = np.array(Y[start:end])
        elapsed = time.time() - t0
        pct = end / N * 100
        print(f'  [{pct:5.1f}%] Wrote samples {start}-{end-1} ({elapsed:.0f}s elapsed)')

print(f'  Total write time: {time.time()-t0:.1f}s')

# Verify
print('Verifying...')
with h5py.File(h5_path, 'r') as f:
    print(f'  X: shape={f[\"X\"].shape}, dtype={f[\"X\"].dtype}')
    print(f'  Y: shape={f[\"Y\"].shape}, dtype={f[\"Y\"].dtype}')
    assert np.allclose(f['X'][0], np.array(X[0])), 'X mismatch!'
    assert np.allclose(f['Y'][0], np.array(Y[0])), 'Y mismatch!'
    assert np.allclose(f['X'][-1], np.array(X[-1])), 'X last mismatch!'
    assert np.allclose(f['Y'][-1], np.array(Y[-1])), 'Y last mismatch!'
    print('  Data integrity check PASSED')

npz_size = os.path.getsize(npz_path)
h5_size = os.path.getsize(h5_path)
print(f'  NPZ size: {npz_size/1e9:.2f} GB')
print(f'  H5 size:  {h5_size/1e9:.2f} GB')
print(f'  NPZ kept intact at {npz_path}')
print('Done!')
"
