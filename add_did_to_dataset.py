import numpy as np
import time
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'data'))
from compute_did import compute_did_features

def process_single(args):
    b, sdf, remaining = args
    mask = sdf <= 0
    did_feat = compute_did_features(mask)
    max_d = np.sqrt(504**2 + 504**2)
    did_feat = np.clip(did_feat / (max_d / 4.0), 0, 1)
    did_feat = did_feat.transpose(2, 0, 1)
    new_x = np.concatenate([did_feat, remaining], axis=0)
    return b, new_x

def main():
    input_path = 'data/augmented/full_dataset.npz'
    output_dir = 'data/augmented_did'
    os.makedirs(output_dir, exist_ok=True)
    
    print("WARNING: Because of the 300GB Lustre disk quota, we cannot pre-allocate an uncompressed 130GB mmap array.")
    print("Instead, we will stream the dataset as individual compressed NPZ chunk files.")
    
    # Remove old bad mmap files if they exist to free up the quota space immediately
    if os.path.exists(os.path.join(output_dir, 'X_mmap.npy')):
        os.remove(os.path.join(output_dir, 'X_mmap.npy'))
    if os.path.exists(os.path.join(output_dir, 'Y_mmap.npy')):
        os.remove(os.path.join(output_dir, 'Y_mmap.npy'))

    print(f"Loading {input_path} with mmap...")
    data = np.load(input_path, mmap_mode='r')
    
    Y = data['Y'] # (N, 1, 504, 504)
    X = data['X'] # (N, 8, 504, 504)
    N = X.shape[0]
    print(f"Total samples: {N}")
    
    chunk_size = 128
    start_time = time.time()
    
    import multiprocessing
    from functools import partial
    num_cpus = multiprocessing.cpu_count()
    print(f"Using {num_cpus} CPUs for parallel DID computation...")
    
    for i in range(0, N, chunk_size):
        end = min(i + chunk_size, N)
        chunk_idx = i // chunk_size
        chunk_file = os.path.join(output_dir, f'chunk_{chunk_idx:04d}.npz')
        
        if os.path.exists(chunk_file):
            print(f"Chunk {chunk_idx:04d} already exists, skipping ({i}-{end})...")
            continue
            
        print(f"Processing chunk {chunk_idx:04d} ({i}-{end} / {N})...")
        
        # Load small chunk into RAM
        X_chunk = np.array(X[i:end])
        Y_chunk = np.array(Y[i:end])
        
        # Index 0 is SDF
        sdf_chunk = X_chunk[:, 0, :, :]
        remaining_features = X_chunk[:, 1:, :, :] # (B, 7, 504, 504)
        B = X_chunk.shape[0]
        
        args_list = [(b, sdf_chunk[b], remaining_features[b]) for b in range(B)]
        
        with multiprocessing.Pool(num_cpus) as pool:
            results = pool.map(process_single, args_list)
            
        new_X_chunk = np.zeros((B, 15, 504, 504), dtype=np.float32)
        for b, new_x in results:
            new_X_chunk[b] = new_x
            
        # Compress and save instantly to disk
        np.savez_compressed(chunk_file, X=new_X_chunk, Y=Y_chunk)
        
        elapsed = time.time() - start_time
        fps = (end - i) / elapsed # FPS of this session
        print(f"Saved {chunk_file} | Speed: {fps:.2f} items/sec")
        start_time = time.time()

    print("Finished processing all chunks!")

if __name__ == '__main__':
    main()
