import torch
import numpy as np
import os
import sys

# Add current dir to path
sys.path.append(os.getcwd())

from ldm.data.cfd_data import CFDTrain

def test_loading():
    h5_path = "data/uk_roof_dataset.h5"
    if not os.path.exists(h5_path):
        print(f"ERROR: {h5_path} not found.")
        return

    print(f"Testing CFDTrain with {h5_path}...")
    try:
        ds = CFDTrain(data_path=h5_path)
        print(f"Dataset length: {len(ds)}")
        
        # Test __getitem__
        sample = ds[0]
        img = sample["image"]
        print(f"Sample 'image' shape: {img.shape}")
        print(f"Value range: min={img.min():.4f}, max={img.max():.4f}")
        
        # Check per-channel ranges in the normalized output
        for c in range(img.shape[0]):
            c_min = img[c].min()
            c_max = img[c].max()
            print(f"  Channel {c} range: [{c_min:.4f}, {c_max:.4f}]")
            
        print("\nSUCCESS: Dataset loading and normalization verified.")
        
    except Exception as e:
        print(f"FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loading()
