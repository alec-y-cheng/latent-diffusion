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

class CFDConditionalDataset(Dataset):
    def __init__(self, data_path, split="train", split_ratio=0.9):
        """
        Custom Dataset for Conditional CFD Generation.
        Target ('image'): 'Y' key (Velocity Magnitude).
        Condition ('cond'): 'X' key (8 Channels: SDF, Inlet Conds, Height, Z, UVW_ref, coords).
        Resizes all inputs to 512x512.
        """
        self.data_path = data_path
        self.split = split
        self.split_ratio = split_ratio
        
        try:
            with np.load(self.data_path) as data:
                # Load Target Y: (N, 1, 504, 504)
                if 'Y' in data:
                    self.y_data = data['Y']
                else:
                    print("Warning: 'Y' key (Target) not found!")
                    self.y_data = np.zeros((10, 1, 504, 504), dtype=np.float32)

                # Load Condition X: (N, 1, 504, 504, 8) -> (N, 8, 504, 504)
                if 'X' in data:
                    raw_x = data['X']
                else:
                    print("Warning: 'X' key (Condition) not found!")
                    raw_x = np.zeros((10, 1, 504, 504, 8), dtype=np.float32)

                if len(raw_x.shape) == 5 and raw_x.shape[1] == 1:
                    raw_x = np.squeeze(raw_x, axis=1) # -> (N, 504, 504, 8)
                
                # Permute X to (N, C, H, W) if needed
                if raw_x.shape[1] != 8 and raw_x.shape[-1] == 8:
                    # Assume (N, H, W, C) -> (N, C, H, W)
                    raw_x = np.transpose(raw_x, (0, 3, 1, 2))
                
                self.x_data = raw_x

        except Exception as e:
            print(f"Error loading {self.data_path}: {e}")
            self.y_data = np.zeros((10, 1, 504, 504), dtype=np.float32)
            self.x_data = np.zeros((10, 8, 504, 504), dtype=np.float32)

        # Ensure lengths match
        assert len(self.y_data) == len(self.x_data), f"Mismatch: Y={len(self.y_data)}, X={len(self.x_data)}"

        num_samples = len(self.y_data)
        split_idx = int(num_samples * self.split_ratio)
        
        # Calculate Separate Normalizations from Training Split
        y_train = self.y_data[:split_idx]
        x_train = self.x_data[:split_idx]

        # Stats for Y (Global min/max usually fine for scalar field)
        self.min_y = np.min(y_train)
        self.max_y = np.max(y_train)
        self.range_y = self.max_y - self.min_y
        if self.range_y == 0: self.range_y = 1e-6

        # Stats for X (Per-Channel min/max)
        self.min_x = np.min(x_train, axis=(0, 2, 3)).reshape(-1, 1, 1)
        self.max_x = np.max(x_train, axis=(0, 2, 3)).reshape(-1, 1, 1)
        self.range_x = self.max_x - self.min_x
        self.range_x[self.range_x == 0] = 1e-6

        if self.split == "train":
            self.y_split = self.y_data[:split_idx]
            self.x_split = self.x_data[:split_idx]
        else:
            self.y_split = self.y_data[split_idx:]
            self.x_split = self.x_data[split_idx:]
        
        print(f"CFDConditional ({self.split}): Samples={len(self.y_split)}")

    def __len__(self):
        return len(self.y_split)

    def __getitem__(self, idx):
        # 1. Process Target (Y)
        y = self.y_split[idx] # (1, 504, 504)
        y = (y - self.min_y) / self.range_y # [0, 1]
        y = y * 2.0 - 1.0 # [-1, 1]
        y = torch.from_numpy(y).float()
        
        # Resize Y to 512x512 (for AE Input)
        y = torch.nn.functional.interpolate(y.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)

        # 2. Process Condition (X)
        x = self.x_split[idx] # (8, 504, 504)
        x = (x - self.min_x) / self.range_x # [0, 1]
        x = x * 2.0 - 1.0 # [-1, 1]
        x = torch.from_numpy(x).float()

        # Resize X to 64x64 (for Concatenation with Latent)
        # Note: We resize directly to matching latent size to save compute (Factor 8)
        cond = torch.nn.functional.interpolate(x.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)

        # Permute to Channels Last (H, W, C) for DDPM compatibility
        y = y.permute(1, 2, 0)
        cond = cond.permute(1, 2, 0)

        return {"image": y, "cond": cond}

class CFDConditionalTrain(CFDConditionalDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, split="train", **kwargs)

class CFDConditionalValidation(CFDConditionalDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, split="validation", **kwargs)
