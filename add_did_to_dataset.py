import numpy as np
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
from compute_did import compute_did_features

def main():
    input_path = 'data/augmented/full_dataset.npz'
    output_dir = 'data/augmented_did'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'full_dataset.npz')

    print(f"Loading {input_path} with mmap...")
    data = np.load(input_path, mmap_mode='r')
    
    Y = data['Y'] # (N, 1, 504, 504)
    X = data['X'] # (N, 8, 504, 504)
    N = X.shape[0]
    print(f"Total samples: {N}")
    
    # We will process in chunks and save locally as we go? Or accumulate?
    # 8.9GB dataset: if we replace 1 channel with 8, X becomes 15 channels.
    # 15/8 = almost double the size. X will be ~15GB.
    # We should probably stream this directly to a new .npz file or process on the fly.
    # Since we must save it, we'll accumulate chunks in a pre-allocated array.
    
    # We will allocate a new numpy array on disk using np.lib.format.open_memmap
    out_x_path = os.path.join(output_dir, 'X_mmap.npy')
    out_y_path = os.path.join(output_dir, 'Y_mmap.npy')
    
    print(f"Pre-allocating out_X with shape {(N, 15, 504, 504)}")
    X_out = np.lib.format.open_memmap(out_x_path, mode='w+', dtype=np.float32, shape=(N, 15, 504, 504))
    print(f"Pre-allocating out_Y with shape {(N, 1, 504, 504)}")
    Y_out = np.lib.format.open_memmap(out_y_path, mode='w+', dtype=np.float32, shape=(N, 1, 504, 504))
    
    chunk_size = 100
    start_time = time.time()
    
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        print(f"Processing chunk {i}-{end} / {N}...")
        
        # Load small chunk into RAM
        X_chunk = np.array(X[i:end])
        Y_chunk = np.array(Y[i:end])
        
        # Index 0 is SDF
        sdf_chunk = X_chunk[:, 0, :, :]
        
        B = X_chunk.shape[0]
        did_list = []
        for b in range(B):
            # Mask generation: sdf <= 0 is obstacle
            mask = sdf_chunk[b] <= 0
            
            # compute_did outputs (H, W, 8)
            did_feat = compute_did_features(mask)
            
            # Heuristic scale from preprocess_did.py
            max_d = np.sqrt(504**2 + 504**2)
            did_feat = np.clip(did_feat / (max_d / 4.0), 0, 1)
            
            # Transpose to (8, H, W)
            did_feat = did_feat.transpose(2, 0, 1)
            did_list.append(did_feat)
            
        did_chunk = np.stack(did_list, axis=0) # (B, 8, 504, 504)
        
        # Keep features 1-7 (drop SDF at 0)
        remaining_features = X_chunk[:, 1:, :, :] # (B, 7, 504, 504)
        
        # Concatenate: DID (8) + Remaining (7) = 15 channels
        new_X_chunk = np.concatenate([did_chunk, remaining_features], axis=1) # (B, 15, 504, 504)
        
        # Save to mmap
        X_out[i:end] = new_X_chunk
        Y_out[i:end] = Y_chunk
        
        elapsed = time.time() - start_time
        fps = end / elapsed
        print(f"Speed: {fps:.2f} items/sec, ETA: {(N-end)/fps:.1f}s")

    print("\nFlushing to disk and saving as NPZ...")
    del X_out, Y_out
    
    # Reload mmap in readonly to save as npz efficiently
    X_final = np.load(out_x_path, mmap_mode='r')
    Y_final = np.load(out_y_path, mmap_mode='r')
    np.savez(output_path, X=X_final, Y=Y_final)
    
    print(f"Saved {output_path} successfully!")
    
    # Clean up massive uncompressed npy arrays
    os.remove(out_x_path)
    os.remove(out_y_path)
    print("Done!")

if __name__ == '__main__':
    main()
