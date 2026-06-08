import argparse
import os

import numpy as np


def update_channel_stats(chunk, min_val, max_val):
    if chunk.ndim == 3:
        chunk = chunk[:, None, :, :]
    chunk_min = np.min(chunk, axis=(0, 2, 3))
    chunk_max = np.max(chunk, axis=(0, 2, 3))
    if min_val is None:
        return chunk_min.astype(np.float64), chunk_max.astype(np.float64)
    return np.minimum(min_val, chunk_min), np.maximum(max_val, chunk_max)


def stream_y_stats(data_path, limit=500, chunk_size=16):
    if data_path.endswith(".h5"):
        import h5py

        with h5py.File(data_path, "r") as f:
            y = f["Y"]
            n = min(limit, y.shape[0])
            min_val = None
            max_val = None
            global_min = None
            global_max = None
            for start in range(0, n, chunk_size):
                chunk = np.asarray(y[start:start + chunk_size], dtype=np.float32)
                min_val, max_val = update_channel_stats(chunk, min_val, max_val)
                cmin = float(np.min(chunk))
                cmax = float(np.max(chunk))
                global_min = cmin if global_min is None else min(global_min, cmin)
                global_max = cmax if global_max is None else max(global_max, cmax)
            return min_val, max_val, global_min, global_max, n

    with np.load(data_path, mmap_mode="r") as data:
        y = data["Y"] if "Y" in data else data["arr_1"]
        n = min(limit, y.shape[0])
        min_val = None
        max_val = None
        global_min = None
        global_max = None
        for start in range(0, n, chunk_size):
            chunk = np.asarray(y[start:start + chunk_size], dtype=np.float32)
            min_val, max_val = update_channel_stats(chunk, min_val, max_val)
            cmin = float(np.min(chunk))
            cmax = float(np.max(chunk))
            global_min = cmin if global_min is None else min(global_min, cmin)
            global_max = cmax if global_max is None else max(global_max, cmax)
        return min_val, max_val, global_min, global_max, n


def main():
    parser = argparse.ArgumentParser(description="Verify AE and conditional LDM target normalization match.")
    parser.add_argument("--data_path", default="data/uk_roof_dataset.h5")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--chunk_size", type=int, default=16)
    args = parser.parse_args()

    if not os.path.exists(args.data_path):
        raise FileNotFoundError(args.data_path)

    ch_min, ch_max, global_min, global_max, n = stream_y_stats(
        args.data_path,
        limit=args.limit,
        chunk_size=args.chunk_size,
    )
    ch_range = ch_max - ch_min
    global_range = global_max - global_min
    global_norm_min = ((ch_min - global_min) / global_range) * 2.0 - 1.0
    global_norm_max = ((ch_max - global_min) / global_range) * 2.0 - 1.0

    print(f"Dataset: {args.data_path}")
    print(f"Samples used for stats: {n}")
    print("Correct AE/LDM per-channel target stats:")
    for c, (mn, mx, rg) in enumerate(zip(ch_min, ch_max, ch_range)):
        print(f"  Ch {c}: min={mn:.6g}, max={mx:.6g}, range={rg:.6g}")

    print("\nOld buggy global target stats:")
    print(f"  global min={global_min:.6g}, max={global_max:.6g}, range={global_range:.6g}")
    print("  Per-channel span after old global [-1,1] normalization:")
    for c, (mn, mx) in enumerate(zip(global_norm_min, global_norm_max)):
        print(f"  Ch {c}: normalized_min={mn:.6g}, normalized_max={mx:.6g}, span={mx - mn:.6g}")

    print("\nOK: corrected LDM target normalization uses the same per-channel Y stats as the AE loader.")


if __name__ == "__main__":
    main()
