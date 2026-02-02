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
        Supports single .npz or directory of .npz files.
        Uses mmap_mode='r' to handle large augmented datasets.
        """
        self.split = split
        self.split_ratio = split_ratio
        
        # 1. Identify Files
        if os.path.isdir(data_path):
            self.files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npz')])
        else:
            self.files = [data_path]
            
        print(f"CFDConditional: Found {len(self.files)} files.")
        
        self.data_chunks_x = []
        self.data_chunks_y = []
        self.chunk_sizes = []
        
        total_samples = 0
        
        # 2. Map Files
        for fpath in self.files:
            try:
                # Use mmap_mode='r' to avoid loading into RAM
                data = np.load(fpath, mmap_mode='r')
                
                # Load Y
                if 'Y' in data:
                    raw_y = data['Y']
                else:
                    raw_y = data['arr_1'] # Fallback?
                    
                # Load X
                if 'X' in data:
                    raw_x = data['X']
                else:
                    raw_x = data['arr_0']

                # Squeeze Singleton (N, 1, 8, ...) -> (N, 8, ...)
                if len(raw_x.shape) == 5 and raw_x.shape[1] == 1:
                    raw_x = raw_x.squeeze(axis=1) # View, efficient
                
                # NO Transpose here if already (N, 8, H, W)
                # If we needed transpose, it would be a view too.
                # Check consistency
                if raw_x.shape[0] != raw_y.shape[0]:
                    print(f"Skipping {fpath}: Size mismatch X={raw_x.shape}, Y={raw_y.shape}")
                    continue
                
                self.data_chunks_x.append(raw_x)
                self.data_chunks_y.append(raw_y)
                self.chunk_sizes.append(raw_x.shape[0])
                total_samples += raw_x.shape[0]
                
            except Exception as e:
                print(f"Error mapping {fpath}: {e}")

        # 3. Stratified Split (Ensure every file contributes to both Train and Val)
        self.indices = []
        
        for i, size in enumerate(self.chunk_sizes):
            # Calculate split point for THIS chunk
            split_point = int(size * self.split_ratio)
            
            if self.split == "train":
                # Add range [0, split_point)
                for local_idx in range(0, split_point):
                    self.indices.append((i, local_idx))
            else:
                # Add range [split_point, size)
                for local_idx in range(split_point, size):
                    self.indices.append((i, local_idx))
            
        self.length = len(self.indices)
        print(f"CFDConditional ({self.split}): Total={self.length} (Stratified across {len(self.files)} files)")

        # 4. Normalization Stats (Compute from FIRST chunk as approximation)
        ref_x = self.data_chunks_x[0]
        ref_y = self.data_chunks_y[0]
        
        limit = min(500, ref_x.shape[0])
        stat_x = ref_x[:limit]
        stat_y = ref_y[:limit]
        
        self.min_y = np.min(stat_y)
        self.max_y = np.max(stat_y)
        self.range_y = self.max_y - self.min_y
        if self.range_y == 0: self.range_y = 1e-6

        self.min_x = np.min(stat_x, axis=(0, 2, 3)).reshape(-1, 1, 1)
        self.max_x = np.max(stat_x, axis=(0, 2, 3)).reshape(-1, 1, 1)
        self.range_x = self.max_x - self.min_x
        self.range_x[self.range_x == 0] = 1e-6

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Retrieve mapped placement
        chunk_idx, local_idx = self.indices[idx]
            
        # Get data
        # Note: mmap access involves disk seek. 
        raw_x = self.data_chunks_x[chunk_idx][local_idx] # (8, 504, 504)
        raw_y = self.data_chunks_y[chunk_idx][local_idx] # (1, 504, 504)
        
        # Convert to float array (loads into RAM)
        x = np.array(raw_x, dtype=np.float32)
        y = np.array(raw_y, dtype=np.float32)
        
        # Normalize
        y = (y - self.min_y) / self.range_y 
        y = y * 2.0 - 1.0 
        y = torch.from_numpy(y).float()
        
        x = (x - self.min_x) / self.range_x
        x = x * 2.0 - 1.0
        x = torch.from_numpy(x).float()

        # Resize Y to 512x512
        y = torch.nn.functional.interpolate(y.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)

        # Resize X to 64x64
        cond = torch.nn.functional.interpolate(x.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)

        # Permute to Channels Last (H, W, C)
        y = y.permute(1, 2, 0)
        cond = cond.permute(1, 2, 0)

        return {"image": y, "cond": cond}

class CFDConditionalTrain(CFDConditionalDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, split="train", **kwargs)

class CFDConditionalValidation(CFDConditionalDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, split="validation", **kwargs)
