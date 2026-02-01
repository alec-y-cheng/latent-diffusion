import torch
from torch.utils.data import Dataset
import numpy as np

class CFDDataset(Dataset):
    def __init__(self, data_path, split="train", split_ratio=0.9):
        """
        Custom Dataset for CFD Velocity Magnitude Fields.
        Expects an .npz file with a 'Y' key containing shape (N, 1, H, W).
        """
        self.data_path = data_path
        self.split = split
        self.split_ratio = split_ratio
        
        try:
            # Determine split index
            # We load the file to get length and stats, 
            # ideally this should be cached or handled more efficiently for huge datasets
            with np.load(self.data_path) as data:
                # Use mmap_mode='r' if the file is very large and uncompressed, 
                # but 'Y' seems to be the key so we access it directly.
                # If wrapped in NpzFile, we might load it into memory.
                # Assuming 2000x1x504x504 floats -> ~2GB, fitting in memory is fine.
                self.full_data = data['Y'] # Copy to memory
                
        except FileNotFoundError:
            print(f"ERROR: Data file not found at {self.data_path}")
            # Create dummy data for initialization check only if file strictly required
            self.full_data = np.zeros((10, 1, 504, 504), dtype=np.float32)

        num_samples = len(self.full_data)
        split_idx = int(num_samples * self.split_ratio)
        
        # Calculate stats for normalization from Training split
        train_data = self.full_data[:split_idx]
        self.min_val = np.min(train_data)
        self.max_val = np.max(train_data)

        if self.split == "train":
            self.data = self.full_data[:split_idx]
        else:
            self.data = self.full_data[split_idx:]
            
        print(f"CFDDataset ({self.split}): Samples={len(self.data)}, Min={self.min_val:.5f}, Max={self.max_val:.5f}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx] # (1, 504, 504)
        
        # Normalize to [-1, 1] for best GAN/Autoencoder performance
        # (x - min) / (max - min) -> [0, 1] -> *2 -1 -> [-1, 1]
        range_val = self.max_val - self.min_val
        if range_val == 0: range_val = 1e-6
        
        x = (x - self.min_val) / range_val
        x = x * 2.0 - 1.0
        
        # Ensure float32
        x = torch.from_numpy(x).float()
        
        # Key 'image' is standard for the LDM codebase
        return {"image": x}

class CFDTrain(CFDDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, split="train", **kwargs)

class CFDValidation(CFDDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, split="validation", **kwargs)
