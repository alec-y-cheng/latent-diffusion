import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

# Configuration
IMG_H, IMG_W = 504, 504
X_CH = 8
Y_CH = 4  # ch0: mag_U, ch1: k, ch2: mag_U_roof, ch3: k_roof

# Canonical Grid Bounds (Analyzed from sample)
X_MIN, X_MAX = -485.0, 521.0
Y_MIN, Y_MAX = -514.5, 491.5
X_STEP, Y_STEP = 2.0, 2.0

def get_canonical_grid():
    # Create the dense grid coordinates for channel 0 and 1
    xs = np.linspace(X_MIN, X_MAX, IMG_W).astype(np.float32)
    ys = np.linspace(Y_MIN, Y_MAX, IMG_H).astype(np.float32)
    xx, yy = np.meshgrid(xs, ys) # (504, 504) each
    return xx, yy

def process_csv(file_path, xx_dense, yy_dense):
    try:
        req_cols = [
            'X', 'Y', 'Z_relative', 'SDF', 'Bldg_height',
            'U_over_Uref', 'dir_sin', 'dir_cos',
            'mag_U', 'k', 'mag_U_roof', 'k_roof'
        ]
        df = pd.read_csv(file_path, usecols=req_cols)

        # Robust Numpy Rasterization
        # 1. Calculate grid indices
        # We round to nearest integer to handle slight float precision differences
        idx_x = np.round((df['X'].values - X_MIN) / X_STEP).astype(int)
        idx_y = np.round((df['Y'].values - Y_MIN) / Y_STEP).astype(int)

        # 2. Filter valid indices (within 0..503)
        valid = (idx_x >= 0) & (idx_x < IMG_W) & (idx_y >= 0) & (idx_y < IMG_H)

        if not np.all(valid):
            # If some points are outside, we strictly filter them out
            # print(f"Warning: {np.sum(~valid)} points out of bounds in {os.path.basename(file_path)}")
            df = df[valid]
            idx_x = idx_x[valid]
            idx_y = idx_y[valid]

        # 3. Initialize Output Arrays (filled with 0.0)
        x_out = np.zeros((X_CH, IMG_H, IMG_W), dtype=np.float32)
        y_out = np.zeros((Y_CH, IMG_H, IMG_W), dtype=np.float32)  # 4 channels

        # 4. Fill Dense Channels (0, 1) - Coordinates
        x_out[0] = xx_dense
        x_out[1] = yy_dense

        # 5. Fill Sparse Channels using advanced indexing
        # Note: idx_y corresponds to rows (H), idx_x to cols (W)

        # Mapping:
        # 2: Z_relative
        x_out[2, idx_y, idx_x] = df['Z_relative'].values
        # 3: SDF
        x_out[3, idx_y, idx_x] = df['SDF'].values
        # 4: Bldg_height
        x_out[4, idx_y, idx_x] = df['Bldg_height'].values
        # 5: U_over_Uref
        x_out[5, idx_y, idx_x] = df['U_over_Uref'].values
        # 6: dir_sin
        x_out[6, idx_y, idx_x] = df['dir_sin'].values
        # 7: dir_cos
        x_out[7, idx_y, idx_x] = df['dir_cos'].values

        # Targets: mag_U (ch0), k (ch1), mag_U_roof (ch2), k_roof (ch3)
        # Guard against OpenFOAM sentinel (-DBL_MAX ~ -1.7976931e+307) leaking
        # through the CSV post-processor as a finite (non-NaN, non-Inf) value.
        # All target quantities are physically non-negative.
        sentinel = 1e100  # anything beyond this magnitude is a sentinel

        mag_u_vals = df['mag_U'].values
        k_vals     = df['k'].values
        mag_u_roof_vals = df['mag_U_roof'].values
        k_roof_vals     = df['k_roof'].values

        mag_u_vals      = np.where(np.abs(mag_u_vals)      > sentinel, 0.0, mag_u_vals)
        k_vals          = np.where(np.abs(k_vals)          > sentinel, 0.0, k_vals)
        mag_u_roof_vals = np.where(np.abs(mag_u_roof_vals) > sentinel, 0.0, mag_u_roof_vals)
        k_roof_vals     = np.where(np.abs(k_roof_vals)     > sentinel, 0.0, k_roof_vals)

        y_out[0, idx_y, idx_x] = np.maximum(mag_u_vals,      0.0)
        y_out[1, idx_y, idx_x] = np.maximum(k_vals,          0.0)
        y_out[2, idx_y, idx_x] = np.maximum(mag_u_roof_vals, 0.0)
        y_out[3, idx_y, idx_x] = np.maximum(k_roof_vals,     0.0)

        return x_out, y_out

    except Exception as e:
        print(f"Failed to process {file_path}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=r"C:\LabShare\Dataset\FormFluxCases\Compressed\Training_Dataset", help="Root directory to search")
    parser.add_argument("--output", type=str, default="new_dataset_uk_roof.npz", help="Output file")
    args = parser.parse_args()

    # 1. Prepare Grid
    print(f"Initializing Grid ({IMG_H}x{IMG_W})...")
    xx, yy = get_canonical_grid()

    # 2. Find Files
    print(f"Searching for CSVs in {args.root}...")
    files = glob.glob(os.path.join(args.root, "**", "*.csv"), recursive=True)
    N_files = len(files)
    print(f"Found {N_files} CSV files.")

    if N_files == 0:
        return

    # 3. Process (Parallel)
    # WINDOWS FIX: Limit to 60 workers max to avoid "WaitMultipleObjects" ValueError (limit is 64)
    max_workers_windows = 60
    n_cores = min(max_workers_windows, int(cpu_count() * 0.9))
    print(f"Processing with {n_cores} workers (Windows Optimized)...")

    # Freeze arguments
    func = partial(process_csv, xx_dense=xx, yy_dense=yy)

    X_list = []
    Y_list = []

    with Pool(n_cores) as p:
        # imap returns an iterator, tqdm wraps it for progress bar
        for x_t, y_t in tqdm(p.imap(func, files), total=N_files):
            if x_t is not None:
                X_list.append(x_t)
                Y_list.append(y_t)

    # 4. Stack and Save
    if len(X_list) > 0:
        print("Stacking...")
        X_final = np.stack(X_list, axis=0).astype(np.float32)
        Y_final = np.stack(Y_list, axis=0).astype(np.float32)

        # [CRITICAL] Clean Data (Inf/NaN -> 0.0)
        print("Cleaning dataset (Inf/NaN -> 0.0) ...")
        X_final = np.nan_to_num(X_final, nan=0.0, posinf=0.0, neginf=0.0)
        Y_final = np.nan_to_num(Y_final, nan=0.0, posinf=0.0, neginf=0.0)


        print(f"Saving to {args.output} (Compressed)...")
        print(f"Final Shapes: X={X_final.shape}, Y={Y_final.shape}")
        np.savez_compressed(args.output, X=X_final, Y=Y_final)
        print("Done!")
    else:
        print("No valid data processed.")

if __name__ == "__main__":
    main()
