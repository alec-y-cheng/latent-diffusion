import torch
import numpy as np
import os
import sys

# Add current dir to path
sys.path.append(os.getcwd())

from ldm.data.cfd_data import CFDTrain

def test_backwards_compatibility():
    # Use one of the old local datasets
    old_path = "data/archive_data/full_dataset.npz"
    if not os.path.exists(old_path):
        print(f"SKIPPING: {old_path} not found. Trying another...")
        # Try finding any npz in data/
        import glob
        candidates = glob.glob("data/*.npz")
        if candidates:
            old_path = candidates[0]
        else:
            print("ERROR: No old .npz found for testing.")
            return

    print(f"Testing CFDTrain with OLD dataset: {old_path}...")
    try:
        ds = CFDTrain(data_path=old_path)
        print(f"Dataset length: {len(ds)}")
        
        # Test __getitem__
        sample = ds[0]
        img = sample["image"]
        print(f"Sample 'image' shape: {img.shape}")
        
        # Verify it's 1-channel if the source is 
        if img.shape[0] == 1:
            print("SUCCESS: Correctly loaded 1-channel data.")
        else:
            print(f"WARNING: Expected 1 channel, got {img.shape[0]}")
            
        print(f"Value range: min={img.min():.4f}, max={img.max():.4f}")
        
        print("\nSUCCESS: Backwards compatibility verified.")
        
    except Exception as e:
        print(f"FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_backwards_compatibility()
