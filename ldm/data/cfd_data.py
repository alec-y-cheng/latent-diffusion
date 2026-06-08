import torch
from torch.utils.data import Dataset
import numpy as np
import os

class CFDDataset(Dataset):
    def __init__(self, data_path, split="train", split_ratio=0.9, size=None):
        """
        Custom Dataset for CFD Velocity Magnitude Fields.
        Supports .npz and .h5 files. 
        Improved for multi-channel support with per-channel normalization.
        """
        self.data_path = data_path
        self.split = split
        self.split_ratio = split_ratio
        self.size = size
        self.is_hdf5 = data_path.endswith('.h5')
        self._h5f = None # Lazy handle for h5py

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Dataset file not found: {self.data_path}")
        
        try:
            if self.is_hdf5:
                import h5py
                with h5py.File(self.data_path, 'r') as f:
                    N = f['Y'].shape[0]
                    self.num_samples = N
                    split_idx = int(N * self.split_ratio)
                    train_limit = min(500, split_idx)
                    # Calculate stats from a training subset
                    train_subset = f['Y'][:train_limit]
                    self.min_val = np.min(train_subset, axis=(0, 2, 3)).reshape(-1, 1, 1)
                    self.max_val = np.max(train_subset, axis=(0, 2, 3)).reshape(-1, 1, 1)
            else:
                # Use mmap_mode to avoid OOM on large NPZs
                with np.load(self.data_path, mmap_mode='r') as data:
                    self.full_data = data['Y']
                    N = self.full_data.shape[0]
                    self.num_samples = N
                    split_idx = int(N * self.split_ratio)
                    train_data = self.full_data[:split_idx]
                    self.min_val = np.min(train_data, axis=(0, 2, 3)).reshape(-1, 1, 1)
                    self.max_val = np.max(train_data, axis=(0, 2, 3)).reshape(-1, 1, 1)
                    
        except Exception as e:
            if self.is_hdf5:
                raise RuntimeError(f"Could not load HDF5 data from {self.data_path}: {e}") from e
            print(f"ERROR: Could not load data from {self.data_path}: {e}")
            self.num_samples = 10
            self.is_hdf5 = False
            self.full_data = np.zeros((10, 1, 504, 504), dtype=np.float32)
            self.min_val = np.array([0.0]).reshape(-1, 1, 1)
            self.max_val = np.array([1.0]).reshape(-1, 1, 1)

        split_idx = int(self.num_samples * self.split_ratio)
        if self.split == "train":
            self.range_indices = (0, split_idx)
            self.length = split_idx
        else:
            self.range_indices = (split_idx, self.num_samples)
            self.length = self.num_samples - split_idx
            
        print(f"CFDDataset ({self.split}): Samples={self.length}, Channels={self.min_val.shape[0]}")
        for c in range(self.min_val.shape[0]):
            print(f"  Ch {c}: Min={self.min_val[c,0,0]:.5f}, Max={self.max_val[c,0,0]:.5f}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # 1. Resolve global index
        global_idx = self.range_indices[0] + idx
        
        # 2. Get Raw Data
        if self.is_hdf5:
            import h5py
            if self._h5f is None:
                self._h5f = h5py.File(self.data_path, 'r')
            y = self._h5f['Y'][global_idx]
        else:
            y = self.full_data[global_idx]
            
        y = np.array(y, dtype=np.float32)
        
        # 3. Per-Channel Normalization to [-1, 1]
        range_val = self.max_val - self.min_val
        range_val[range_val == 0] = 1e-6
        
        y = (y - self.min_val) / range_val
        y = y * 2.0 - 1.0
        
        # 4. Convert to Tensor
        y_tensor = torch.from_numpy(y).float()
        
        if self.size is not None:
            y_tensor = torch.nn.functional.interpolate(
                y_tensor.unsqueeze(0), 
                size=(self.size, self.size), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
            
        return {"image": y_tensor}

class CFDTrain(CFDDataset):
    def __init__(self, data_path, size=None, **kwargs):
        super().__init__(data_path=data_path, split="train", size=size, **kwargs)

class CFDValidation(CFDDataset):
    def __init__(self, data_path, size=None, **kwargs):
        super().__init__(data_path=data_path, split="validation", size=size, **kwargs)

class CFDConditionalDataset(Dataset):
    @staticmethod
    def _channel_min_max(data):
        if data.ndim == 3:
            data = data[:, None, :, :]
        if data.ndim != 4:
            raise ValueError(f"Expected target data with shape (N,C,H,W) or (N,H,W), got {data.shape}")
        min_val = np.min(data, axis=(0, 2, 3)).reshape(-1, 1, 1)
        max_val = np.max(data, axis=(0, 2, 3)).reshape(-1, 1, 1)
        return min_val, max_val

    def _set_y_stats(self, stat_y):
        self.min_y, self.max_y = self._channel_min_max(stat_y)
        self.range_y = self.max_y - self.min_y
        self.range_y[self.range_y == 0] = 1e-6

    def _print_y_stats(self):
        print("CFDConditional target normalization: per-channel [-1, 1]")
        for c in range(self.min_y.shape[0]):
            print(f"  Y Ch {c}: Min={self.min_y[c,0,0]:.5f}, Max={self.max_y[c,0,0]:.5f}")

    def __init__(self, data_path, split="train", split_ratio=0.9, augment=False):
        """
        Custom Dataset for Conditional CFD Generation.
        Supports single .npz or directory of .npz files.
        Uses mmap_mode='r' to handle large augmented datasets.
        """
        self.split = split
        self.split_ratio = split_ratio
        self.augment = augment

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset path not found: {data_path}")
        
        # 1. Identify Files
        if os.path.isdir(data_path):
            self.files = sorted([os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.npz') or f.endswith('.h5')])
        else:
            self.files = [data_path]

        if not self.files:
            raise FileNotFoundError(f"No .npz or .h5 files found in dataset path: {data_path}")
            
        print(f"CFDConditional: Found {len(self.files)} files.")
        
        self.data_chunks_x = []
        self.data_chunks_y = []
        self.chunk_sizes = []
        
        total_samples = 0
        
        self.is_hdf5 = False
        if len(self.files) == 1 and self.files[0].endswith('.h5'):
            self.is_hdf5 = True
            import h5py
            with h5py.File(self.files[0], 'r') as h5f:
                total_samples = h5f['X'].shape[0]
                split_point = int(total_samples * self.split_ratio)
                
                # Match CFDDataset/autoencoder target stats: per-channel stats
                # from the first training subset.
                limit = min(500, split_point if split_point > 0 else total_samples)
                stat_x = h5f['X'][:limit]
                stat_y = h5f['Y'][:limit]
                
                self._set_y_stats(stat_y)
                self.min_x = np.min(stat_x, axis=(0, 2, 3)).reshape(-1, 1, 1)
                self.max_x = np.max(stat_x, axis=(0, 2, 3)).reshape(-1, 1, 1)
                self.range_x = self.max_x - self.min_x
                self.range_x[self.range_x == 0] = 1e-6
            
            # Setup indices
            if self.split == "train":
                self.indices = [(0, i) for i in range(0, split_point)]
            else:
                self.indices = [(0, i) for i in range(split_point, total_samples)]
            
            self.length = len(self.indices)
            print(f"CFDConditional HDF5 ({self.split}): Total={self.length}")
            self._print_y_stats()
            self._h5f = None # Open lazily per worker
        else:
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
            
            ref_split_point = int(ref_x.shape[0] * self.split_ratio)
            limit = min(500, ref_split_point if ref_split_point > 0 else ref_x.shape[0])
            stat_x = ref_x[:limit]
            stat_y = ref_y[:limit]
            
            self._set_y_stats(stat_y)
    
            self.min_x = np.min(stat_x, axis=(0, 2, 3)).reshape(-1, 1, 1)
            self.max_x = np.max(stat_x, axis=(0, 2, 3)).reshape(-1, 1, 1)
            self.range_x = self.max_x - self.min_x
            self.range_x[self.range_x == 0] = 1e-6
            self._print_y_stats()
    
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Retrieve mapped placement
        chunk_idx, local_idx = self.indices[idx]
            
        # Get data
        if getattr(self, 'is_hdf5', False):
            import h5py
            if self._h5f is None:
                self._h5f = h5py.File(self.files[0], 'r')
            raw_x = self._h5f['X'][local_idx]
            raw_y = self._h5f['Y'][local_idx]
        else:
            # Note: mmap access involves disk seek. 
            raw_x = self.data_chunks_x[chunk_idx][local_idx] # (8, 504, 504)
            raw_y = self.data_chunks_y[chunk_idx][local_idx] # (1, 504, 504)
        
        # Convert to float array (loads into RAM)
        x = np.array(raw_x, dtype=np.float32)
        y = np.array(raw_y, dtype=np.float32)

        # Augmentation (On-the-fly)
        if self.augment:
            # 1. Transpose: Swap spatial dims (H, W) -> (W, H)
            # This allows constructing 90-degree rotations (Transpose + Flip)
            if np.random.rand() < 0.5:
                x = np.swapaxes(x, 1, 2) # (C, W, H)
                y = np.swapaxes(y, 1, 2) # (1, W, H)
                
                x_copy = x.copy()
                if x.shape[0] == 15:
                    # DID mapping: X<->Y (11<->12), Sin<->Cos (13<->14)
                    x[11], x[12] = x_copy[12], x_copy[11]
                    x[13], x[14] = x_copy[14], x_copy[13]
                    # Swap DID compass across diagonal: East(0)<->South(2), West(4)<->North(6), SW(3)<->NE(7)
                    x[0], x[2] = x_copy[2], x_copy[0]
                    x[4], x[6] = x_copy[6], x_copy[4]
                    x[3], x[7] = x_copy[7], x_copy[3]
                else:
                    x[4], x[5] = x_copy[5], x_copy[4]
                    x[7], x[6] = x_copy[6], x_copy[7]
                del x_copy

            # 2. Random Horizontal Flip (p=0.5)
            if np.random.rand() < 0.5:
                x = x[:, :, ::-1]
                y = y[:, :, ::-1]
                
                if x.shape[0] == 15:
                    x[11] *= -1 # Invert X_local
                    x[13] *= -1 # Invert Sin (X-component)
                    # Horizontal swap: East(0)<->West(4), SE(1)<->SW(3), NE(7)<->NW(5)
                    x_copy = x.copy()
                    x[0], x[4] = x_copy[4], x_copy[0]
                    x[1], x[3] = x_copy[3], x_copy[1]
                    x[5], x[7] = x_copy[7], x_copy[5]
                    del x_copy
                else:
                    x[4] *= -1
                    x[6] *= -1

            # 3. Random Vertical Flip (p=0.5)
            if np.random.rand() < 0.5:
                x = x[:, ::-1, :]
                y = y[:, ::-1, :]
                
                if x.shape[0] == 15:
                    x[12] *= -1 # Invert Y_local
                    x[14] *= -1 # Invert Cos (Y-component)
                    # Vertical flip: South(2)<->North(6), SE(1)<->NE(7), SW(3)<->NW(5)
                    x_copy = x.copy()
                    x[2], x[6] = x_copy[6], x_copy[2]
                    x[1], x[7] = x_copy[7], x_copy[1]
                    x[3], x[5] = x_copy[5], x_copy[3]
                    del x_copy
                else:
                    x[5] *= -1
                    x[7] *= -1
                
            # Handle negative strides from flipping for torch compatibility
            if x.strides[1] < 0 or x.strides[2] < 0:
                 x = x.copy()
            if y.strides[1] < 0 or y.strides[2] < 0:
                 y = y.copy()
        
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
    def __init__(self, data_path, augment=True, **kwargs):
        super().__init__(data_path=data_path, split="train", augment=augment, **kwargs)

class CFDConditionalValidation(CFDConditionalDataset):
    def __init__(self, data_path, **kwargs):
        super().__init__(data_path=data_path, split="validation", **kwargs)
