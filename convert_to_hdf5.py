import os
import glob
import numpy as np
import h5py
from tqdm import tqdm

def main():
    input_dir = 'data/augmented_did'
    output_file = 'data/augmented_did_h5/dataset.h5'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    chunk_files = sorted(glob.glob(os.path.join(input_dir, 'chunk_*.npz')))
    if not chunk_files:
        print("No chunk files found!")
        return
        
    print(f"Found {len(chunk_files)} chunk files. Creating HDF5 structure...")
    
    # We will compute total N
    total_chunks = len(chunk_files)
    total_N = 0
    
    # Fast shape check
    shape_x = None
    shape_y = None
    for f in chunk_files:
        data = np.load(f)
        x = data['X']
        if shape_x is None:
            shape_x = x.shape[1:]
            shape_y = data['Y'].shape[1:]
        total_N += x.shape[0]

    with h5py.File(output_file, 'w') as h5f:
        # Create datasets with chunking for fast random access and GZIP compression
        # We chunk them at the sample level: e.g. (1, 15, 504, 504)
        chunk_size_x = (1,) + shape_x
        chunk_size_y = (1,) + shape_y
        
        dset_x = h5f.create_dataset('X', shape=(total_N,) + shape_x, dtype=np.float32, 
                                    chunks=chunk_size_x, compression='gzip')
        dset_y = h5f.create_dataset('Y', shape=(total_N,) + shape_y, dtype=np.float32, 
                                    chunks=chunk_size_y, compression='gzip')
                                    
        current_idx = 0
        for f in tqdm(chunk_files, desc="Converting to HDF5"):
            data = np.load(f)
            x_arr = data['X']
            y_arr = data['Y']
            n = x_arr.shape[0]
            
            dset_x[current_idx:current_idx+n] = x_arr
            dset_y[current_idx:current_idx+n] = y_arr
            
            current_idx += n
            
    print(f"Successfully created heavily-compressed HDF5 dataset at {output_file}!")

if __name__ == '__main__':
    main()
