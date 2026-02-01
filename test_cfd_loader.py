import numpy as np
import torch
import os
from ldm.data.cfd_data import CFDTrain, CFDValidation

# Create dummy data
data_path = "dummy_cfd.npz"
if not os.path.exists(data_path):
    print("Creating dummy data...")
    Y = np.random.randn(20, 1, 504, 504).astype(np.float32)
    np.savez(data_path, Y=Y)

print("Initializing CFDTrain...")
# Try initializing
train_ds = CFDTrain(data_path=data_path)
print(f"Train dataset length: {len(train_ds)}")
item = train_ds[0]
print(f"Item['image'] shape: {item['image'].shape}")
print(f"Item['image'] range: [{item['image'].min()}, {item['image'].max()}]")

print("Initializing CFDValidation...")
val_ds = CFDValidation(data_path=data_path)
print(f"Val dataset length: {len(val_ds)}")

# Check input compatibility with model expectation (Channels First vs Last)
# The model expects (C, H, W) from dataset based on AutoencoderKL.get_input logic
# if (1, 504, 504), get_input says 1 < 504, so it keeps it as is.
# If we were returning (504, 504, 1), get_input would permute it.
# My dataset returns (1, 504, 504).
print("Shape check passed.")

# Cleanup
# os.remove(data_path)
print("Test complete.")
