import torch
from torch.utils.data import Dataset
import numpy as np
import os

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
    def __init__(self, data_path, split="train", split_ratio=0.9, augment=False):
        """
        Custom Dataset for Conditional CFD Generation.
        Supports single .npz or directory.
        Uses mmap_mode='r' to handle large datasets.
        """
        self.split = split
        self.split_ratio = split_ratio
        self.augment = augment  # Only augment if True (Training)
        
        # 1. Identify Files
        if os.path.isdir(data_path):
            # Look for ANY .npz or data_orig_X.npy or just accept the folder
            # If user passes 'data/augmented', we might find multiple parts.
            # If user passes 'data/full_dataset.npz', handle that too.
            # Let's support both:
            files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npz')]
            npy_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('_X.npy')]
        elif os.path.isfile(data_path):
             files = [data_path]
             npy_files = []
        else:
             files = []
             npy_files = []

        self.data_sources = [] # List of (X_mmap, Y_mmap)
        self.sample_indices = [] # List of (source_idx, sample_idx)
        
        # Helper to load
        def load_source(fpath):
            try:
                if fpath.endswith('.npz'):
                    data = np.load(fpath, mmap_mode='r')
                    if 'X' in data and 'Y' in data:
                        return data['X'], data['Y']
                    elif 'arr_0' in data: # Fallback
                        return data['arr_0'], data['arr_1'] if 'arr_1' in data else None
                elif fpath.endswith('_X.npy'):
                    y_path = fpath.replace('_X.npy', '_Y.npy')
                    if os.path.exists(y_path):
                         X = np.load(fpath, mmap_mode='r')
                         Y = np.load(y_path, mmap_mode='r')
                         return X, Y
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
            return None, None

        # Load all sources
        sources_to_check = files + npy_files
        print(f"Dataset ({split}) source path: '{data_path}'")
        print(f"Dataset ({split}): Found {len(sources_to_check)} potential files.")
        
        for f in sources_to_check:
            X, Y = load_source(f)
            if X is not None and Y is not None:
                # Handle singleton
                if X.ndim == 5 and X.shape[1] == 1:
                    pass # We will index [i, 0] later or wrapper handles it? 
                    # Actually mmap slicing is tricky if we reshape. 
                    # Let's just store as is and handle in __getitem__
                    pass
                
                # Check lengths
                n_samples = X.shape[0]
                source_idx = len(self.data_sources)
                self.data_sources.append((X, Y))
                
                # Split indices
                split_point = int(n_samples * self.split_ratio)
                if self.split == "train":
                    indices = range(0, split_point)
                else:
                    indices = range(split_point, n_samples)
                    
                for i in indices:
                    self.sample_indices.append((source_idx, i))

        print(f"Dataset ({split}): Total Samples={len(self.sample_indices)} (Augment={self.augment})")
        
        # Stats (Approximate from first source)
        if len(self.data_sources) > 0:
            X0, Y0 = self.data_sources[0]
            # Just take first few
            sampX = X0[:min(100, len(X0))]
            if sampX.ndim == 5: sampX = sampX[:, 0]
            
            sampY = Y0[:min(100, len(Y0))]
            
            self.min_val = np.min(sampY)
            self.max_val = np.max(sampY)
            self.range_y = self.max_val - self.min_val
            if self.range_y == 0: self.range_y = 1e-6
            
            # X stats per channel or global? 
            # Global usually safer for stability unless channels very different
            self.min_x = np.min(sampX) # Simple global min/max for now
            self.max_x = np.max(sampX)
            self.range_x = self.max_x - self.min_x
            if self.range_x == 0: self.range_x = 1e-6
        else:
            self.min_val, self.max_val, self.range_y = -1, 1, 2
            self.min_x, self.max_x, self.range_x = -1, 1, 2

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        source_idx, local_idx = self.sample_indices[idx]
        X_source, Y_source = self.data_sources[source_idx]
        
        # Load raw data (this reads from disk)
        raw_x = np.array(X_source[local_idx]) # (C, H, W) or (1, C, H, W)
        raw_y = np.array(Y_source[local_idx]) # (1, H, W)
        
        if raw_x.ndim == 4: raw_x = raw_x[0] # Squeeze singleton (1, 8, 504, 504) -> (8, 504, 504)
        if raw_y.ndim == 4: raw_y = raw_y[0]
        
        # Augmentation (On-the-fly)
        if self.augment:
            # 1. Flip H
            if np.random.rand() < 0.5:
                raw_x = raw_x[:, :, ::-1]
                raw_y = raw_y[:, :, ::-1]
                # Invert X-components and Cos (Flow X direction)
                # Indices: 4 (X_local), 7 (Cos)
                raw_x[4] *= -1
                raw_x[7] *= -1

            # 2. Flip V
            if np.random.rand() < 0.5:
                raw_x = raw_x[:, ::-1, :]
                raw_y = raw_y[:, ::-1, :]
                # Invert Y-components and Sin (Flow Y direction)
                # Indices: 5 (Y_local), 6 (Sin)
                raw_x[5] *= -1
                raw_x[6] *= -1
                
            # 3. Rotate 90 (Optional, can replace FlipHV)
            # if np.random.rand() < 0.5: ... (handling vector rotation is tricky, stick to flips for now as they cover 4 quadrants)

        # Convert to Tensor
        x = torch.from_numpy(raw_x.copy()).float()
        y = torch.from_numpy(raw_y.copy()).float()
        
        # Normalize
        y = (y - self.min_val) / self.range_y
        y = y * 2.0 - 1.0
        
        x = (x - self.min_x) / self.range_x
        x = x * 2.0 - 1.0
        
        # Resize
        # Y -> 512 (Model Output)
        # X -> 64  (Conditioning)
        y = torch.nn.functional.interpolate(y.unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0)
        cond = torch.nn.functional.interpolate(x.unsqueeze(0), size=(64, 64), mode='bilinear', align_corners=False).squeeze(0)
        
        # Permute to Channels Last (H, W, C)
        y = y.permute(1, 2, 0)
        cond = cond.permute(1, 2, 0)
        
        return {"image": y, "cond": cond}

class CFDConditionalTrain(CFDConditionalDataset):
    def __init__(self, data_path, augment=True, **kwargs):
        super().__init__(data_path=data_path, split="train", augment=augment, **kwargs)

class CFDConditionalValidation(CFDConditionalDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, split="validation", augment=False, **kwargs)
