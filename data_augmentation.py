import numpy as np
import os
import argparse
from tqdm import tqdm
import sys

def augment_dataset(input_path, output_dir):
    print(f"Loading original dataset from {input_path}...")
    if not os.path.exists(input_path):
        print(f"ERROR: File not found at {input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)

    try:
        # Load Data
        # mmap_mode='r' allows loading metadata without full read, but for processing we might need RAM.
        # Given X=24GB, we can load it if we process sequentially.
        with np.load(input_path) as data:
            if 'X' in data:
                X_orig = data['X']
            else:
                X_orig = data['arr_0'] # Fallback
            
            if 'Y' in data:
                Y_orig = data['Y']
            else:
                # If Y missing, maybe it's self-supervised? But user has Y.
                print("Warning: Y not found, using dummy.")
                Y_orig = np.zeros((len(X_orig), 1, 504, 504))

        print(f"Loaded X: {X_orig.shape}, Y: {Y_orig.shape}")
        
        # Define Transformations
        transforms = ["orig", "flip_h", "flip_v", "flip_hv"]
        
        # Channel Mapping for Vector Inversion
        # User X channels: ['SDF' 'Bldg_height' 'Z_relative' 'U_over_Uref' 'X_local' 'Y_local' 'dir_sin' 'dir_cos']
        # Indices: 0, 1, 2, 3, 4, 6, 7 (Wait, indices are 0-7)
        # 4: X_local, 5: Y_local? (User list has 8 items)
        # 6: dir_sin, 7: dir_cos
        
        # Let's assume standard indices based on user provided list:
        # 0: SDF
        # 1: Bldg_h
        # 2: Z_rel
        # 3: U_over_Uref
        # 4: X_local
        # 5: Y_local
        # 6: dir_sin
        # 7: dir_cos
        
        CH_X = 4
        CH_Y = 5
        CH_SIN = 6
        CH_COS = 7

        for mode in transforms:
            print(f"Processing mode: {mode}...")
            
            # Create Copy to modify
            X_new = X_orig.copy()
            Y_new = Y_orig.copy()
            
            # CAUTION: X shape is (N, 1, 8, 504, 504) or (N, 8, 504, 504)?
            # User said: (2000, 1, 8, 504, 504).
            # We need to handle that singleton dim 1.
            
            has_singleton = False
            if len(X_new.shape) == 5 and X_new.shape[1] == 1:
                has_singleton = True
                X_new = X_new.squeeze(1) # (N, 8, H, W)
                
            # Now X_new is (N, C, H, W)
            # Y_new is (N, 1, H, W)

            if mode == "orig":
                pass
            
            elif mode == "flip_h":
                # Horizontal Flip (Mirror W -> last dim)
                X_new = X_new[..., ::-1]
                Y_new = Y_new[..., ::-1]
                
                # Invert X-components
                # X_local (4) and Cos (7)
                X_new[:, CH_X, :, :] *= -1
                X_new[:, CH_COS, :, :] *= -1
                
            elif mode == "flip_v":
                # Vertical Flip (Mirror H -> second last dim)
                X_new = X_new[..., ::-1, :]
                Y_new = Y_new[..., ::-1, :]
                
                # Invert Y-components
                # Y_local (5) and Sin (6)
                X_new[:, CH_Y, :, :] *= -1
                X_new[:, CH_SIN, :, :] *= -1
                
            elif mode == "flip_hv":
                # Rotate 180 (Flip both)
                X_new = X_new[..., ::-1, ::-1]
                Y_new = Y_new[..., ::-1, ::-1]
                
                # Invert BOTH X and Y components
                X_new[:, CH_X, :, :] *= -1
                X_new[:, CH_COS, :, :] *= -1
                X_new[:, CH_Y, :, :] *= -1
                X_new[:, CH_SIN, :, :] *= -1

            # Restore singleton if needed
            if has_singleton:
                X_new = np.expand_dims(X_new, axis=1)

            # Save as separate .npy files to allow mmap
            base_name = os.path.join(output_dir, f"data_{mode}")
            print(f"Saving {base_name}_X.npy / _Y.npy ...")
            
            np.save(f"{base_name}_X.npy", X_new)
            np.save(f"{base_name}_Y.npy", Y_new)
            
            # Free memory
            del X_new, Y_new

    except Exception as e:
        print(f"CRASH: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to original .npz")
    parser.add_argument("--output", type=str, required=True, help="Directory to save parts")
    args = parser.parse_args()
    
    augment_dataset(args.input, args.output)