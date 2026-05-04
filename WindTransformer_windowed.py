# =========================
# Conditional Transformer Diffusion (DDPM/DDIM) with EMA + memmap loading (Windows-safe)
# - Avoids RAM blowups by extracting X.npy / Y.npy from .npz and using mmap
# - Train-only normalization computed streaming (low RAM)
# - AMP training + grad clipping
# - EMA weights for sampling
# - DDIM (eta=0) sampler (correct drop-in)
# - Save-only plot (Agg backend) with *shared color scale* for honest comparison


import os
import sys
import json
os.environ["MPLBACKEND"] = "Agg"          # must be before importing pyplot
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # pragmatic workaround for Windows libiomp conflicts

import math
import random
import shutil
import zipfile
import numpy as np
import torch
from tqdm import tqdm
torch.set_num_threads(1)




import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from contextlib import nullcontext

# -------------------------
# Extra knobs
# -------------------------
USE_ADALN_TIME = True

USE_REL_POS_BIAS = True
REL_POS_BIAS_PER_HEAD = False   
ATTN_MASK_NEG = -1e4            # large negative instead of -inf for fp16 stability

X0_AUX_WEIGHT = 0.01            # Reduced from 0.1 to prevent numerical drift at high precision

# --- Gradient Matching Regularization ---
USE_GRAD_LOSS = True                # Toggle gradient-matching loss
GRAD_LOSS_WEIGHT = 1e-4             # Weight for gradient loss (1e-4 to 1e-3)

# -------------------------
# Config
# -------------------------
SEED = 63
USE_AUGMENTED_DATA = False # Toggle to use the 8,000 sample dataset

if USE_AUGMENTED_DATA:
    # Try local Windows path first (for testing), then cluster path
    LOCAL_PATH = "expanded_dataset_aug.npz"
    CLUSTER_PATH = "/home/hice1/ikaradag3/scratch/Conditional_Transformer/dataset/expanded_dataset_aug.npz"
    
    if os.path.exists(LOCAL_PATH):
        DATA_PATH = LOCAL_PATH
    else:
        DATA_PATH = CLUSTER_PATH
        
    print(f"!!! CONFIG: Using EXPLICIT AUGMENTED DATASET !!!")
    print(f"!!! Path: {DATA_PATH} !!!")
else:
    DATA_PATH = "/home/hice1/ikaradag3/scratch/Conditional_Transformer/dataset/expanded_dataset.npz"
    print(f"CONFIG: Using Standard Dataset without Physical Augmentation")

# -------------------------
# Output Directories (Defaults, can be overridden by CLI)
# -------------------------
OUT_DIR = "results"
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
PLOT_DIR = os.path.join(OUT_DIR, "plots")
IMG_DIR  = os.path.join(OUT_DIR, "images")

def update_dirs(new_out_dir):
    global OUT_DIR, CKPT_DIR, PLOT_DIR, IMG_DIR, cache_dir, stats_path
    OUT_DIR = new_out_dir
    CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
    PLOT_DIR = os.path.join(OUT_DIR, "plots")
    IMG_DIR  = os.path.join(OUT_DIR, "images")
    
    # Isolate cache per experiment to avoid interference
    dataset_slug = os.path.splitext(os.path.basename(DATA_PATH))[0]
    # [USER REQUEST] Use a common cache folder to save disk space
    cache_dir = os.path.join(os.path.dirname(DATA_PATH), f"_npy_cache_{dataset_slug}_results")
    stats_path = os.path.join(cache_dir, "stats.pt")

    os.makedirs(CKPT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    os.makedirs(IMG_DIR, exist_ok=True)
    print(f"Outputs will be saved to: {OUT_DIR}")
    print(f"Isolated Data Cache: {cache_dir}")

IMG_H, IMG_W = 504, 504
X_CH = 8
Y_CH = 1


###############################
# Model params (architecture) #
###############################
# Patch size MUST divide 504 exactly (e.g. 6, 7, 8, 9, 12, 14, 18)
PATCH = 9        # Reverted to Source (56x56 tokens) for maximum detail
EMB = 512        # Reverted to 512 for efficiency
HEADS = 8        # Reverted to 8 (512/64)
DEPTH = 20       # Reverted to 20
MLP_RATIO = 4.0
DROPOUT = 0.02
###############################
###############################

# --- Windowed attention (big speedup for small PATCH) ---
ATTN_MODE = "window"     # {"window", "global"}
SHIFT_WINDOWS = True     # Swin-style shift every other block (with proper mask)

def pick_window(ph, preferred=(9, 8, 7, 14, 16, 28)):
    for w in preferred:
        if w <= ph and ph % w == 0:
            return w
    for w in range(min(ph, 14), 1, -1):
        if ph % w == 0:
            return w
    return ph

PH, PW, WINDOW_SIZE = 0, 0, 0
def recalculate_window_params(patch_size, silent=False):
    global PH, PW, WINDOW_SIZE
    PH = IMG_H // patch_size
    PW = IMG_W // patch_size
    WINDOW_SIZE = pick_window(PH)
    if not silent:
        print(f"[window] PATCH={patch_size} -> ph=pw={PH}, using WINDOW_SIZE={WINDOW_SIZE}")

# Initialize with defaults
recalculate_window_params(PATCH, silent=True)

T = 1000
BETA_START = 1e-4
BETA_END   = 2e-2

BATCH_SIZE = 8
LR = 1e-4 # Reverted to baseline
WEIGHT_DECAY = 1e-4
EPOCHS = 200 

NUM_WORKERS = 8
PIN_MEMORY = True       
PERSISTENT_WORKERS = False 

SAMPLE_STEPS = 200              # Full quality for final inference
PLOT_STEPS = 20                # Compromise: Faster than 50, better quality than 15
EMA_DECAY = 0.9995              # Smoother output, matching source

STATS_BATCH = 1         
EPS_STD = 1e-6
gammad = 0.99           # Smoother decay (Was 0.99)
lamda = 0.25            # L1 loss weight (Original default)

split = 0.9             # fraction of data used for training

# Loss objective: "eps" is standard DDPM.
LOSS_OBJECTIVE = "eps"  # {"eps"}

# --- Augmentation Config (Offline only) ---
AUG_CH_X = 0            # Horizontal/X coordinate
AUG_CH_Y = 1            # Vertical/Y coordinate
AUG_CH_DIR_SIN = 6      # Vertical/Y component
AUG_CH_DIR_COS = 7      # Horizontal/X component

# Define Config Dictionary for Reference/Saving
CONFIG = {
    "PATCH": PATCH,
    "EMB": EMB,
    "HEADS": HEADS,
    "DEPTH": DEPTH,
    "ATTN_MODE": ATTN_MODE,
    "WINDOW_SIZE": WINDOW_SIZE,
    "LR": LR,
    "BATCH_SIZE": BATCH_SIZE,
    "EPOCHS": EPOCHS,
    "EMA_DECAY": EMA_DECAY,
    "LAMBDA": lamda,
    "GAMMA": gammad,
    "DATA_PATH": DATA_PATH,
    "OUT_DIR": OUT_DIR,
    "AUG_CH_DIR_SIN": AUG_CH_DIR_SIN,
    "AUG_CH_DIR_COS": AUG_CH_DIR_COS,
}

# Refiner options (helps remove patch seams without changing output resolution)
USE_REFINER = True
REFINER_USE_YT = True   # include y_t in refiner input
REFINER_USE_X  = True   # include x_cond (compressed) in refiner input
REFINER_X_PROJ_CH = 4   # channels after 1x1 projection of x_cond

# Optional compile (PyTorch 2.x). Can help, can also hurt on some setups.
# DISABLED: Currently causing 140GB+ VRAM usage spikes on H200
USE_TORCH_COMPILE = False

# -------------------------
# Repro
# -------------------------
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    # Enable TF32 for significant speedup on Hopper (H200) Tensor Cores
    torch.set_float32_matmul_precision('high')
use_amp = (device.type == "cuda")
print("Device:", device)

# -------------------------
# Torch backend knobs
# -------------------------
if device.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(False)
    except Exception:
        pass

# -------------------------
# Extract X.npy / Y.npy from .npz without loading into RAM
# -------------------------
def extract_npz_members(npz_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. List members using Python (fast) to see what we have
    with zipfile.ZipFile(npz_path, "r", allowZip64=True) as zf:
        all_names = zf.namelist()
        
    x_names = sorted([n for n in all_names if n.startswith("X") and n.endswith(".npy")])
    y_names = sorted([n for n in all_names if n.startswith("Y") and n.endswith(".npy")])
    
    if not x_names:
        # Fallback for old single-file datasets
        if "X.npy" in all_names: x_names = ["X.npy"]
        
    if not y_names:
         if "Y.npy" in all_names: y_names = ["Y.npy"]

    print(f"Found X chunks: {x_names}")
    print(f"Found Y chunks: {y_names}")
    sys.stdout.flush()

    x_paths = []
    y_paths = []

    for xn in x_names:
        out_p = os.path.join(out_dir, xn)
        x_paths.append(out_p)
        if not os.path.exists(out_p):
            # Atomic extraction with size verification
            try:
                with zipfile.ZipFile(npz_path, "r", allowZip64=True) as zf:
                    info = zf.getinfo(xn)
                    expected_size = info.file_size
                    
                    tmp_p = out_p + ".tmp"
                    print(f"Extracting {xn} (expected size: {expected_size/1e9:.2f} GB)...")
                    sys.stdout.flush()
                    
                    with zf.open(xn) as source, open(tmp_p, "wb") as target:
                        shutil.copyfileobj(source, target)
                        target.flush()
                        os.fsync(target.fileno())
                    
                    actual_size = os.path.getsize(tmp_p)
                    if actual_size != expected_size:
                        raise IOError(f"Size mismatch for {xn}: expected {expected_size}, got {actual_size}. Likely DISK QUOTA EXCEEDED.")
                    
                    os.replace(tmp_p, out_p) # Atomic rename
                    print(f"Successfully extracted {xn}.")
            except Exception as e:
                if 'tmp_p' in locals() and os.path.exists(tmp_p): os.remove(tmp_p)
                print(f"!!! ERROR: Failed to extract {xn}: {e} !!!")
                raise e

    for yn in y_names:
        out_p = os.path.join(out_dir, yn)
        y_paths.append(out_p)
        if not os.path.exists(out_p):
            try:
                with zipfile.ZipFile(npz_path, "r", allowZip64=True) as zf:
                    info = zf.getinfo(yn)
                    expected_size = info.file_size
                    
                    tmp_p = out_p + ".tmp"
                    print(f"Extracting {yn} (expected size: {expected_size/1e9:.2f} GB)...")
                    sys.stdout.flush()
                    
                    with zf.open(yn) as source, open(tmp_p, "wb") as target:
                        shutil.copyfileobj(source, target)
                        target.flush()
                        os.fsync(target.fileno())
                    
                    actual_size = os.path.getsize(tmp_p)
                    if actual_size != expected_size:
                        raise IOError(f"Size mismatch for {yn}: expected {expected_size}, got {actual_size}. Likely DISK QUOTA EXCEEDED.")
                        
                    os.replace(tmp_p, out_p) # Atomic rename
                    print(f"Successfully extracted {yn}.")
            except Exception as e:
                if 'tmp_p' in locals() and os.path.exists(tmp_p): os.remove(tmp_p)
                print(f"!!! ERROR: Failed to extract {yn}: {e} !!!")
                raise e

    return x_paths, y_paths

# -------------------------
# Support "Early" Restart Config (Overriding DATA_PATH before cache calculation)
# -------------------------
restart_conf_path = os.path.join("Restart_Enabled", "restart_config.py")
if os.path.exists(restart_conf_path):
    print(f"!!! FOUND EARLY RESTART CONFIG: {restart_conf_path} !!!")
    import importlib.util
    spec = importlib.util.spec_from_file_location("restart_config", restart_conf_path)
    restart_params = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(restart_params)

    # Override Globals
    if hasattr(restart_params, 'DATA_PATH'):
        DATA_PATH = restart_params.DATA_PATH
        print(f"!!! OVERRIDE DATA_PATH -> {DATA_PATH} !!!")
    if hasattr(restart_params, 'LR'):
        LR = restart_params.LR
        print(f"!!! OVERRIDE LR -> {LR} !!!")
    if hasattr(restart_params, 'gammad'):
        gammad = restart_params.gammad
        print(f"!!! OVERRIDE GAMMA -> {gammad} !!!")
    if hasattr(restart_params, 'EPOCHS'):
        EPOCHS = restart_params.EPOCHS
# -------------------------

# -------------------------
# Cache config (Isolated in update_dirs)
# -------------------------
dataset_slug = os.path.splitext(os.path.basename(DATA_PATH))[0]
cache_dir = os.path.join(os.path.dirname(DATA_PATH), f"_npy_cache_{dataset_slug}_results")
stats_path = os.path.join(cache_dir, "stats.pt")

# --- Global placeholders for inference/tests ---
X_mm = None
Y_mm = None
val_idx = None
train_idx = None

def get_memmap_data(override_ram=None):
    """Utility to load/extract memmapped data without full training pass."""
    global X_mm, Y_mm, val_idx, train_idx
    if X_mm is not None:
        return X_mm, Y_mm, train_idx, val_idx
        
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
        
    # Define Virtual Concatenation Class
    class VirtualConcatenatedArray:
        def __init__(self, arrays):
            self.arrays = arrays
            self.starts = np.cumsum([0] + [a.shape[0] for a in arrays[:-1]])
            self.ends = np.cumsum([a.shape[0] for a in arrays])
            self.shape = (self.ends[-1], *arrays[0].shape[1:])
            self.dtype = arrays[0].dtype

        def __getitem__(self, idx):
            if isinstance(idx, int):
                if idx < 0: idx += self.shape[0]
                if idx >= self.shape[0] or idx < 0: raise IndexError(f"Index {idx} out of bounds")
                
                # Binary search or linear (linear is fast enough for <10 chunks)
                for i, (start, end) in enumerate(zip(self.starts, self.ends)):
                    if start <= idx < end:
                        return self.arrays[i][idx - start]
            elif isinstance(idx, slice):
                 # Not strictly needed for this training loop but nice to have
                 raise NotImplementedError("Slicing not fully supported on VirtualArray")
            else:
                 raise TypeError(f"Invalid index type: {type(idx)}")

    try:
        x_paths, y_paths = extract_npz_members(DATA_PATH, cache_dir)
        
        # [OPTIMIZATION] Load to RAM defaults to True, but can be overridden
        if override_ram is not None:
            USE_RAM = override_ram
        else:
            # DEFAULT: True (Load into RAM for speed)
            # We upped Slurm request to 250G, so 75GB fits comfortably.
            USE_RAM = True 
            if os.name == 'nt':
                USE_RAM = False # Default to mmap on Windows to be safe

        # Helper to load with or without mmap
        def load_arrays(paths, use_ram):
            arrays = []
            for p in paths:
                max_retries = 2
                for attempt in range(max_retries):
                    try:
                        mode = None if use_ram else "r"
                        arr = np.load(p, mmap_mode=mode)
                        # Force a read check for truncated file
                        _ = arr.shape
                        if use_ram:
                            print(f"Loaded {os.path.basename(p)} into RAM.")
                        else:
                            print(f"Mapped {os.path.basename(p)} (mmap_mode='r').")
                        arrays.append(arr)
                        break # Success
                    except (ValueError, EOFError, OSError, RuntimeError) as e:
                        print(f"!!! CACHE CORRUPTION DETECTED in {p} (Attempt {attempt+1}/{max_retries}): {e} !!!")
                        if os.path.exists(p): os.remove(p)
                        if attempt < max_retries - 1:
                            print(f"!!! Retrying extraction for {p}... !!!")
                            extract_npz_members(DATA_PATH, cache_dir)
                        else:
                            raise RuntimeError(f"Permanent corruption or Disk Quota error for {p}. Please check disk space.")
            return arrays
            
        X_parts = load_arrays(x_paths, USE_RAM)
        Y_parts = load_arrays(y_paths, USE_RAM)
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize data/loaders: {e}")
        # If it was a corruption error, the user just needs to re-run.
        # Otherwise, likely disk space.
        sys.exit(1)
            
    if len(X_parts) == 1:
        X_mm = X_parts[0]
    else:
        print(f"Concatenating {len(X_parts)} X chunks virtually.")
        X_mm = VirtualConcatenatedArray(X_parts)

    if len(Y_parts) == 1:
        Y_mm = Y_parts[0]
    else:
        print(f"Concatenating {len(Y_parts)} Y chunks virtually.")
        Y_mm = VirtualConcatenatedArray(Y_parts)
    
    N = int(X_mm.shape[0])
    
    # Leakage-prevention split for 4x augmented dataset (Original, H, V, HV)
    # Leakage-prevention split for augmented dataset (Original, H, V [, HV])
    # Detect augmentation factor (3x or 4x supported)
    if USE_AUGMENTED_DATA:
        # Check standard factors (Prioritize 3x as it is our current standard)
        if N % 3 == 0:
            AUG_FACTOR = 3
        elif N % 4 == 0:
            AUG_FACTOR = 4
        else:
            print(f"WARNING: Dataset size {N} is not divisible by 3 or 4. Falling back to standard random split (Risk of leakage!)")
            AUG_FACTOR = 1
            
        if AUG_FACTOR > 1:
            N_base = N // AUG_FACTOR
            print(f"Detected {AUG_FACTOR}x Augmentation. Base samples: {N_base}")
            
            perm_base = np.random.RandomState(SEED).permutation(N_base)
            n_train_base = max(1, int(split * N_base))
            train_base = perm_base[:n_train_base]
            val_base = perm_base[n_train_base:] if (N_base - n_train_base) > 0 else perm_base[:1]
            
            # Training gets all augmented versions of the base training samples
            train_list = [train_base + i * N_base for i in range(AUG_FACTOR)]
            train_idx = np.concatenate(train_list)
            
            # Validation stays pure (only original samples)
            val_idx = val_base
            print(f"Leakage-prevention split: {len(train_idx)} train (incl. augmentations), {len(val_idx)} val (base only)")
        else:
            # Fallback for non-divisible
            perm = np.random.RandomState(SEED).permutation(N)
            n_train = max(1, int(split * N))
            train_idx = perm[:n_train]
            val_idx = perm[n_train:]
    else:
        # Standard split (for non-augmented or custom counts)
        perm = np.random.RandomState(SEED).permutation(N)
        n_train = max(1, int(split * N))
        train_idx = perm[:n_train]
        val_idx = perm[n_train:] if (N - n_train) > 0 else perm[:1]
        print(f"Standard split: {len(train_idx)} train, {len(val_idx)} val")
    
    print(f"Initialized memmaps: X={X_mm.shape}, Y={Y_mm.shape}")
    return X_mm, Y_mm, train_idx, val_idx

# -------------------------
# Dataset reading from memmap
# -------------------------

# -------------------------
# Dataset reading from memmap
# -------------------------
class XYMemmapDataset(Dataset):
    def __init__(self, X_mm, Y_mm, indices, x_ch=8, img_hw=(504, 504), augment=False):
        self.X_mm = X_mm
        self.Y_mm = Y_mm
        self.indices = np.asarray(indices, dtype=np.int64)
        self.x_ch = int(x_ch)
        self.H, self.W = img_hw
        self.augment = bool(augment)

    def __len__(self):
        return int(self.indices.shape[0])

    def __getitem__(self, i):
        idx = int(self.indices[i])
        x = self.X_mm[idx]
        y = self.Y_mm[idx]

        # X: (1,8,H,W) -> (8,H,W)
        if x.ndim == 4 and x.shape[0] == 1 and x.shape[1] == self.x_ch:
            x = x[0]
        # Y: (H,W) -> (1,H,W)
        if y.ndim == 2 and y.shape[0] == self.H and y.shape[1] == self.W:
            y = y[None, ...]
        elif y.ndim == 3 and y.shape[0] == 1 and y.shape[1] == self.H and y.shape[2] == self.W:
            pass
        else:
            raise ValueError(f"Unexpected Y item shape {y.shape}")

        # Fix: duplicate array to ensure it is writable and contiguous, fixing the warning
        x = np.array(x, copy=True)
        y = np.array(y, copy=True)

        if self.augment:
            # 1. Random Horizontal Flip (mirroring around vertical axis)
            if random.random() < 0.5:
                # Mirror spatial dimensions (W)
                x = x[:, :, ::-1].copy()
                y = y[:, :, ::-1].copy()
                # Vectors: x-component (sin) must be negated
                # dir_sin is at AUG_CH_DIR_SIN (6)
                x[AUG_CH_DIR_SIN] = -x[AUG_CH_DIR_SIN]
                # Coordinates: x-component (Channel 0) must be negated
                # We assume AUG_CH_X is 0
                x[AUG_CH_X] = -x[AUG_CH_X]

            # 2. Random Vertical Flip (mirroring around horizontal axis)
            if random.random() < 0.5:
                # Mirror spatial dimensions (H)
                x = x[:, ::-1, :].copy()
                y = y[:, ::-1, :].copy()
                # Vectors: y-component (cos) must be negated
                # dir_cos is at AUG_CH_DIR_COS (7)
                x[AUG_CH_DIR_COS] = -x[AUG_CH_DIR_COS]
                # Coordinates: y-component (Channel 1) must be negated
                # We assume AUG_CH_Y is 1
                x[AUG_CH_Y] = -x[AUG_CH_Y]

        x_t = torch.from_numpy(x).to(torch.float32)
        y_t = torch.from_numpy(y).to(torch.float32)

        # [FIX] Inf/NaN cleaning is now done in convert_to_npz.py, so we skip runtime check for speed
        # x_t = torch.nan_to_num(x_t, nan=0.0, posinf=0.0, neginf=0.0)
        # y_t = torch.nan_to_num(y_t, nan=0.0, posinf=0.0, neginf=0.0)

        return x_t, y_t


# Dataset and Loader placeholders (Global)
train_loader = None
val_loader = None
x_mean = None
x_std = None
y_mean = None
y_std = None

# -------------------------
# Spatial Gradient Computation (for gradient-matching loss)
# -------------------------
# Pre-defined Sobel kernels (3x3, more robust to noise than simple [−1,0,1])
SOBEL_X = torch.tensor([[[[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]]]], dtype=torch.float32) / 8.0  # Normalized

SOBEL_Y = torch.tensor([[[[-1, -2, -1],
                          [ 0,  0,  0],
                          [ 1,  2,  1]]]], dtype=torch.float32) / 8.0  # Normalized


def compute_spatial_gradients(field, mode="simple"):
    """Compute ∂f/∂x and ∂f/∂y using finite differences.
    
    Args:
        field: (B, 1, H, W) tensor
        mode: "simple" for [-1, 0, 1]/2 or "sobel" for 3x3 Sobel kernels
    Returns:
        grad_x, grad_y: (B, 1, H, W) tensors
    """
    if mode == "sobel":
        # Sobel is more robust to noise (uses 3x3 smoothing)
        kx = SOBEL_X.to(field.dtype).to(field.device)
        ky = SOBEL_Y.to(field.dtype).to(field.device)
        grad_x = F.conv2d(field, kx, padding=1)
        grad_y = F.conv2d(field, ky, padding=1)
    else:
        # Simple [-1, 0, 1] / 2 kernels for x and y gradients
        kx = torch.tensor([[[[-1.0, 0.0, 1.0]]]], dtype=field.dtype, device=field.device) / 2.0
        ky = kx.transpose(-1, -2)
        grad_x = F.conv2d(field, kx, padding=(0, 1))
        grad_y = F.conv2d(field, ky, padding=(1, 0))
    return grad_x, grad_y

# -------------------------
# Compute train-only mean/std streaming
# -------------------------
@torch.no_grad()
def compute_mean_std(train_loader_stats):
    x_sum   = torch.zeros((1, X_CH, 1, 1), dtype=torch.float64)
    x_sumsq = torch.zeros((1, X_CH, 1, 1), dtype=torch.float64)
    x_count = 0

    y_sum   = torch.zeros((1, 1, 1, 1), dtype=torch.float64)
    y_sumsq = torch.zeros((1, 1, 1, 1), dtype=torch.float64)
    y_count = 0

    for xb, yb in tqdm(train_loader_stats, desc="Computing Stats"):
        xb64 = xb.to(torch.float64)
        yb64 = yb.to(torch.float64)

        B = int(xb64.shape[0])
        x_count += B * IMG_H * IMG_W
        y_count += B * IMG_H * IMG_W

        x_sum   += xb64.sum(dim=(0, 2, 3), keepdim=True)
        x_sumsq += (xb64 * xb64).sum(dim=(0, 2, 3), keepdim=True)

        y_sum   += yb64.sum(dim=(0, 2, 3), keepdim=True)
        y_sumsq += (yb64 * yb64).sum(dim=(0, 2, 3), keepdim=True)

    x_mean64 = x_sum / float(x_count)
    x_var64  = (x_sumsq / float(x_count)) - (x_mean64 * x_mean64)
    x_mean = x_mean64.to(torch.float32)
    x_std  = torch.sqrt(torch.clamp(x_var64.to(torch.float32), min=EPS_STD))

    y_mean64 = y_sum / float(y_count)
    y_var64  = (y_sumsq / float(y_count)) - (y_mean64 * y_mean64)
    y_mean = y_mean64.to(torch.float32)
    y_std  = torch.sqrt(torch.clamp(y_var64.to(torch.float32), min=EPS_STD))

    return x_mean, x_std, y_mean, y_std

# Initialize global stats placeholders with defaults
x_mean = torch.zeros((1, X_CH, 1, 1), device=device)
x_std  = torch.ones((1, X_CH, 1, 1), device=device)
y_mean = torch.zeros((1, Y_CH, 1, 1), device=device)
y_std  = torch.ones((1, Y_CH, 1, 1), device=device)

def load_or_compute_stats(dataset):
    global x_mean, x_std, y_mean, y_std
    if os.path.exists(stats_path):
        print(f"Loading cached stats from {stats_path} ...")
        stats = torch.load(stats_path, map_location=device)
        x_mean, x_std = stats["x_mean"].to(device), stats["x_std"].to(device)
        y_mean, y_std = stats["y_mean"].to(device), stats["y_std"].to(device)
    else:
        print(f"Computing dataset mean/std (using batch_{STATS_BATCH} and {NUM_WORKERS} workers)...")
        train_loader_stats = DataLoader(dataset, batch_size=STATS_BATCH, shuffle=False, num_workers=NUM_WORKERS, drop_last=False)
        x_mean_cpu, x_std_cpu, y_mean_cpu, y_std_cpu = compute_mean_std(train_loader_stats)
        torch.save({
            "x_mean": x_mean_cpu, "x_std": x_std_cpu,
            "y_mean": y_mean_cpu, "y_std": y_std_cpu
        }, stats_path)
        print(f"Saved stats to {stats_path}")
        x_mean, x_std = x_mean_cpu.to(device), x_std_cpu.to(device)
        y_mean, y_std = y_mean_cpu.to(device), y_std_cpu.to(device)

def norm_x(x): return (x - x_mean) / x_std
def norm_y(y): return (y - y_mean) / y_std
def denorm_y(y): return y * y_std + y_mean

print("Computed stats:",
      "x_mean", tuple(x_mean.shape), "x_std", tuple(x_std.shape),
      "y_mean", tuple(y_mean.shape), "y_std", tuple(y_std.shape))

# -------------------------
# Diffusion schedule (DDPM)
# -------------------------
betas = torch.linspace(BETA_START, BETA_END, T, dtype=torch.float32)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)  # ᾱ_t

sqrt_alpha_bars = torch.sqrt(alpha_bars)
sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

def extract(a_1d, t, x_shape):
    out = a_1d.to(t.device).gather(0, t).float()
    return out.view(-1, *([1] * (len(x_shape) - 1)))

def q_sample(y0, t, noise):
    sa = extract(sqrt_alpha_bars, t, y0.shape)
    so = extract(sqrt_one_minus_alpha_bars, t, y0.shape)
    return sa * y0 + so * noise

# -------------------------
# Model utilities
# -------------------------
def sinusoidal_timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(0, half, dtype=torch.float32, device=t.device) / half)
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

def _to_grid(tok, ph, pw):
    B, N, C = tok.shape
    assert N == ph * pw, f"N={N} != ph*pw={ph*pw}"
    return tok.view(B, ph, pw, C)

def _to_tokens(grid):
    B, ph, pw, C = grid.shape
    return grid.view(B, ph * pw, C)

def window_partition(grid, win):
    # grid: (B, ph, pw, C) -> (B*nwin, win*win, C)
    B, ph, pw, C = grid.shape
    assert ph % win == 0 and pw % win == 0, "ph/pw must be divisible by window size"
    grid = grid.view(B, ph // win, win, pw // win, win, C)
    grid = grid.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, nh, nw, win, win, C)
    return grid.view(B * (ph // win) * (pw // win), win * win, C)

def window_unpartition(windows, ph, pw, win, B):
    # windows: (B*nwin, win*win, C) -> (B, ph, pw, C)
    _, _, C = windows.shape
    nh = ph // win
    nw = pw // win
    windows = windows.view(B, nh, nw, win, win, C)
    windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(B, ph, pw, C)

def build_swin_attn_mask(ph, pw, win, shift, device, neg=ATTN_MASK_NEG):
    """
    Swin attention mask for shifted windows.
    Returns (nwin, M, M) float32 with 0 allowed and neg disallowed.
    """
    if shift == 0:
        return None

    img_mask = torch.zeros((1, ph, pw, 1), device=device, dtype=torch.int64)
    cnt = 0
    h_slices = (slice(0, -win), slice(-win, -shift), slice(-shift, None))
    w_slices = (slice(0, -win), slice(-win, -shift), slice(-shift, None))
    for hs in h_slices:
        for ws in w_slices:
            img_mask[:, hs, ws, :] = cnt
            cnt += 1

    mask_windows = window_partition(img_mask, win)               # (nwin, M, 1)
    mask_windows = mask_windows.view(-1, win * win)              # (nwin, M)
    attn = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # (nwin, M, M)

    attn = attn.ne(0).to(torch.float32) * float(neg)             # 0 or neg
    return attn.clone()                                          # (nwin, M, M)

# -------------------------
# Attention blocks
# -------------------------
_HAS_SDPA = hasattr(F, "scaled_dot_product_attention")

class SDPA_Attention(nn.Module):
    def __init__(self, dim, heads, dropout=0.0):
        super().__init__()
        if not _HAS_SDPA:
            raise RuntimeError("torch.nn.functional.scaled_dot_product_attention not found. Use PyTorch 2.0+.")
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.dropout = float(dropout)
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim, bias=True)

    def forward(self, x, attn_mask=None):
        # x: (B, L, C)
        B, L, C = x.shape
        qkv = self.qkv(x)  # (B, L, 3C)
        qkv = qkv.view(B, L, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)  # (3, B, h, L, d)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, h, L, d)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,  # None or broadcastable to (B, h, L, L)
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # (B, h, L, d)

        out = out.transpose(1, 2).contiguous().view(B, L, C)  # (B, L, C)
        return self.proj(out)

class GlobalTransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0, use_adaln_time=True):
        super().__init__()
        self.use_adaln_time = bool(use_adaln_time)

        self.norm1 = nn.LayerNorm(dim, elementwise_affine=not self.use_adaln_time)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=not self.use_adaln_time)

        if self.use_adaln_time:
            self.ada1 = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
            self.ada2 = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))

        self.attn = SDPA_Attention(dim, heads, dropout=dropout)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def _apply_adaln(self, x, temb, ada, norm):
        h = norm(x)
        if not self.use_adaln_time:
            return h
        scale, shift = ada(temb).chunk(2, dim=-1)
        return h * (1.0 + scale[:, None, :]) + shift[:, None, :]

    def forward(self, x, temb):
        h1 = self._apply_adaln(x, temb, self.ada1, self.norm1) if self.use_adaln_time else self.norm1(x)
        x = x + self.attn(h1)

        h2 = self._apply_adaln(x, temb, self.ada2, self.norm2) if self.use_adaln_time else self.norm2(x)
        x = x + self.mlp(h2)
        return x

class WindowTransformerBlock(nn.Module):
    def __init__(
        self,
        dim, heads, ph, pw,
        win=8, shift=0,
        mlp_ratio=4.0, dropout=0.0,
        use_adaln_time=True,
        use_rel_pos_bias=True,
        rel_pos_bias_per_head=False,
    ):
        super().__init__()
        self.ph, self.pw = int(ph), int(pw)
        self.win = int(win)
        self.shift = int(shift)
        self.dim = int(dim)
        self.heads = int(heads)

        assert self.ph % self.win == 0 and self.pw % self.win == 0, "ph/pw must be divisible by window size"
        assert 0 <= self.shift < self.win, "shift must be in [0, win)"

        self.use_adaln_time = bool(use_adaln_time)
        self.use_rel_pos_bias = bool(use_rel_pos_bias)
        self.rel_pos_bias_per_head = bool(rel_pos_bias_per_head)

        # AdaLN (time) – disable LN affine when AdaLN is used
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=not self.use_adaln_time)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=not self.use_adaln_time)
        if self.use_adaln_time:
            self.ada1 = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
            self.ada2 = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
            # Zero-init adaptive scaling for stability (DiT approach)
            nn.init.zeros_(self.ada1[-1].weight)
            nn.init.zeros_(self.ada1[-1].bias)
            nn.init.zeros_(self.ada2[-1].weight)
            nn.init.zeros_(self.ada2[-1].bias)

        self.attn = SDPA_Attention(dim, heads, dropout=dropout)

        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

        # shifted-window mask (lazy per device)
        # shifted-window mask (pre-calculated to satisfy torch.compile/CUDAGraphs)
        if self.shift > 0:
            mask = build_swin_attn_mask(
                self.ph, self.pw, self.win, self.shift, "cpu", neg=ATTN_MASK_NEG
            )
            self.register_buffer("_shift_mask", mask, persistent=False)
        else:
            self.register_buffer("_shift_mask", None, persistent=False)

        # Relative position bias (Swin-style)
        if self.use_rel_pos_bias:
            M = self.win * self.win
            coords_h = torch.arange(self.win)
            coords_w = torch.arange(self.win)
            coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # (2, win, win)
            coords_flat = coords.flatten(1)                                          # (2, M)

            rel = coords_flat[:, :, None] - coords_flat[:, None, :]                  # (2, M, M)
            rel = rel.permute(1, 2, 0).contiguous()                                  # (M, M, 2)
            rel[:, :, 0] += self.win - 1
            rel[:, :, 1] += self.win - 1
            rel[:, :, 0] *= (2 * self.win - 1)
            rel_index = rel.sum(-1)                                                  # (M, M)
            self.register_buffer("rel_pos_index", rel_index, persistent=False)

            table_size = (2 * self.win - 1) * (2 * self.win - 1)
            if self.rel_pos_bias_per_head:
                self.rel_pos_bias_table = nn.Parameter(torch.zeros(table_size, self.heads))
            else:
                self.rel_pos_bias_table = nn.Parameter(torch.zeros(table_size, 1))
            nn.init.trunc_normal_(self.rel_pos_bias_table, std=0.02)

    def _get_shift_mask(self, device):
        if self.shift == 0:
            return None
        return self._shift_mask # already pre-calculated and on the right device via register_buffer

    def _get_rel_pos_bias(self, device):
        # returns (1, 1, M, M) float32 (shared) or (1, heads, M, M) float32 (per-head)
        if not self.use_rel_pos_bias:
            return None
        M = self.win * self.win
        idx = self.rel_pos_index.view(-1)  # (M*M,)
        bias = self.rel_pos_bias_table[idx].view(M, M, -1)  # (M, M, 1 or heads)
        bias = bias.permute(2, 0, 1).contiguous()           # (1 or heads, M, M)
        bias = bias.unsqueeze(0)                            # (1, 1/heads, M, M)
        return bias.to(device=device, dtype=torch.float32)

    def _apply_adaln(self, x, temb, ada, norm):
        h = norm(x)
        if not self.use_adaln_time:
            return h
        scale, shift = ada(temb).chunk(2, dim=-1)  # (B, C) each
        return h * (1.0 + scale[:, None, :]) + shift[:, None, :]

    def forward(self, x, temb):
        # x: (B, N, C), temb: (B, C)
        B, N, C = x.shape
        assert N == self.ph * self.pw, "Token count mismatch"

        # --- attention branch ---
        h = self._apply_adaln(x, temb, self.ada1, self.norm1) if self.use_adaln_time else self.norm1(x)
        grid = _to_grid(h, self.ph, self.pw)  # (B, ph, pw, C)

        if self.shift > 0:
            grid = torch.roll(grid, shifts=(-self.shift, -self.shift), dims=(1, 2))

        xw = window_partition(grid, self.win)  # (B*nwin, M, C)
        Bnwin = xw.shape[0]
        M = self.win * self.win

        # --- build additive mask (always 4D when present) ---
        shift_mask = self._get_shift_mask(xw.device)            # (nwin, M, M) or None
        rel_bias   = self._get_rel_pos_bias(xw.device)          # (1,1,M,M) or (1,heads,M,M) or None

        attn_mask = None
        if shift_mask is not None:
            # (nwin,M,M) -> (B*nwin,1,M,M)
            attn_mask = shift_mask.repeat(B, 1, 1).unsqueeze(1)  # float32, (Bnwin,1,M,M)

        if rel_bias is not None:
            if self.rel_pos_bias_per_head:
                # rel_bias: (1,heads,M,M)
                if attn_mask is None:
                    attn_mask = rel_bias.expand(Bnwin, -1, -1, -1)  # (Bnwin,heads,M,M)
                else:
                    attn_mask = attn_mask.expand(-1, self.heads, -1, -1) + rel_bias  # (Bnwin,heads,M,M)
            else:
                # rel_bias: (1,1,M,M)
                if attn_mask is None:
                    attn_mask = rel_bias.expand(Bnwin, -1, -1, -1)  # (Bnwin,1,M,M)
                else:
                    attn_mask = attn_mask + rel_bias                # (Bnwin,1,M,M)

        # --- FIX #1: correct residual form (no xw = xw + attn(xw)) ---
        attn_out = self.attn(xw, attn_mask=attn_mask)               # (Bnwin, M, C)
        grid_attn = window_unpartition(attn_out, self.ph, self.pw, self.win, B)

        if self.shift > 0:
            grid_attn = torch.roll(grid_attn, shifts=(self.shift, self.shift), dims=(1, 2))

        x = x + _to_tokens(grid_attn)

        # --- MLP branch ---
        h2 = self._apply_adaln(x, temb, self.ada2, self.norm2) if self.use_adaln_time else self.norm2(x)
        x = x + self.mlp(h2)
        return x

# -------------------------
# Patch Transformer Denoiser
# -------------------------
class PatchTransformerDenoiser(nn.Module):
    def __init__(
        self,
        in_ch=9,
        patch=12,
        emb=512,
        depth=10,
        heads=8,
        mlp_ratio=4.0,
        dropout=0.05,
        img_hw=(504, 504),
        attn_mode="window",
        window_size=8,
        shift_windows=True,
        use_refiner=True,
        refiner_use_yt=True,
        refiner_use_x=True,
        refiner_x_proj_ch=4,
    ):
        super().__init__()
        self.patch = int(patch)
        self.H, self.W = img_hw
        assert self.H % self.patch == 0 and self.W % self.patch == 0, "Image size must be divisible by patch size."
        self.ph = self.H // self.patch
        self.pw = self.W // self.patch
        self.n_tokens = self.ph * self.pw
        self.emb = int(emb)

        self.attn_mode = str(attn_mode).lower()
        self.window_size = int(window_size)
        self.shift_windows = bool(shift_windows)

        if self.attn_mode == "window":
            assert self.ph % self.window_size == 0 and self.pw % self.window_size == 0, (
                f"WINDOW_SIZE={self.window_size} must divide ph={self.ph}, pw={self.pw}."
            )

        self.in_ch = int(in_ch)
        self.cond_ch = self.in_ch - 1  # x_cond channels (since y_t is 1ch)

        # Patchify -> tokens
        self.patch_embed = nn.Conv2d(self.in_ch, self.emb, kernel_size=self.patch, stride=self.patch, bias=True)

        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, self.emb))
        nn.init.trunc_normal_(self.pos, std=0.02)

        self.time_mlp = nn.Sequential(
            nn.Linear(self.emb, self.emb * 4),
            nn.GELU(),
            nn.Linear(self.emb * 4, self.emb),
        )

        blocks = []
        for i in range(int(depth)):
            if self.attn_mode == "global":
                blk = GlobalTransformerBlock(
                    self.emb, heads,
                    mlp_ratio=mlp_ratio, dropout=dropout,
                    use_adaln_time=USE_ADALN_TIME
                )
            elif self.attn_mode == "window":
                shift = (self.window_size // 2) if (self.shift_windows and (i % 2 == 1)) else 0
                blk = WindowTransformerBlock(
                    self.emb, heads,
                    ph=self.ph, pw=self.pw,
                    win=self.window_size,
                    shift=shift,
                    mlp_ratio=mlp_ratio,
                    dropout=dropout,
                    use_adaln_time=USE_ADALN_TIME,
                    use_rel_pos_bias=USE_REL_POS_BIAS,
                    rel_pos_bias_per_head=REL_POS_BIAS_PER_HEAD,
                )
            else:
                raise ValueError(f"Unknown attn_mode={self.attn_mode}")
            blocks.append(blk)

        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(self.emb)

        # Predict per-token P*P pixels (then unpatchify to full-res)
        self.head = nn.Linear(self.emb, self.patch * self.patch)

        # Optional pixel-space refiner
        self.use_refiner = bool(use_refiner)
        self.refiner_use_yt = bool(refiner_use_yt)
        self.refiner_use_x = bool(refiner_use_x)

        if self.use_refiner:
            in_ref_ch = 1  # eps
            if self.refiner_use_yt:
                in_ref_ch += 1  # y_t
            if self.refiner_use_x:
                self.x_proj = nn.Conv2d(self.cond_ch, refiner_x_proj_ch, kernel_size=1)
                in_ref_ch += refiner_x_proj_ch

            self.refine = nn.Sequential(
                nn.Conv2d(in_ref_ch, 64, 3, padding=1), nn.GELU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.GELU(),
                nn.Conv2d(64, 1, 3, padding=1),
            )

    def unpatchify(self, patches):
        # patches: (B, N, P*P) -> (B,1,H,W)
        B, N, PP = patches.shape
        P = self.patch
        patches = patches.view(B, self.ph, self.pw, P, P)         # (B,ph,pw,P,P)
        patches = patches.permute(0, 1, 3, 2, 4).contiguous()     # (B,ph,P,pw,P)
        img = patches.view(B, 1, self.H, self.W)
        return img

    def forward(self, x_cond, y_t, t):
        inp = torch.cat([x_cond, y_t], dim=1)                     # (B,9,H,W)

        tok = self.patch_embed(inp)                               # (B,emb,ph,pw)
        tok = tok.flatten(2).transpose(1, 2)                      # (B,N,emb)
        tok = tok + self.pos

        t_emb = sinusoidal_timestep_embedding(t, self.emb)        # (B, emb)
        temb = self.time_mlp(t_emb)                               # (B, emb)

        for blk in self.blocks:
            tok = blk(tok, temb)

        tok = self.norm(tok)

        eps_patches = self.head(tok)                              # (B,N,P*P)
        eps = self.unpatchify(eps_patches)                        # (B,1,H,W)

        if self.use_refiner:
            ref_parts = [eps]
            if self.refiner_use_yt:
                ref_parts.append(y_t.to(dtype=eps.dtype))
            if self.refiner_use_x:
                ref_parts.append(self.x_proj(x_cond.to(dtype=eps.dtype)))
            ref_in = torch.cat(ref_parts, dim=1)
            eps = eps + self.refine(ref_in)                       # residual correction

        return eps

# -------------------------
# EMA
# -------------------------
class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    @torch.no_grad()
    def to(self, dev):
        for k in list(self.shadow.keys()):
            self.shadow[k] = self.shadow[k].to(dev)

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            self.backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name].data)

    @torch.no_grad()
    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name in self.backup:
                p.data.copy_(self.backup[name].data)
        self.backup = {}

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

# Note: Model, EMA, and Optimizer moved to __main__ block below to ensure CLI overrides work correctly.

# -------------------------
# Train / Val
# -------------------------
def run_epoch(loader, train: bool):
    model.train(train)
    total, count = 0.0, 0

    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp) if use_amp else nullcontext()
    grad_ctx = torch.enable_grad() if train else torch.no_grad()

    # Add tqdm progress bar
    pbar = tqdm(loader, desc="Train" if train else "Val", leave=False)

    # Manually enter context to avoid indenting the entire loop body
    torch.set_grad_enabled(train)

    for xb_idx, (xb, yb) in enumerate(pbar):
        if train and device.type == "cuda":
            # Helps CUDAGraphs/torch.compile manage dependencies between batches
            torch.compiler.cudagraph_mark_step_begin()

        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        xb = norm_x(xb)
        yb = norm_y(yb)

        B = xb.shape[0]
        t = torch.randint(0, T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(yb)
        y_t = q_sample(yb, t, noise)

        with amp_ctx:
            eps_pred = model(xb, y_t, t)

            # --- eps loss with wake-weighting ---
            if LOSS_OBJECTIVE == "eps":
                # Pixel-wise loss
                l2_px = F.mse_loss(eps_pred, noise, reduction='none')
                l1_px = F.l1_loss(eps_pred, noise, reduction='none')
                eps_loss_px = (1 - lamda) * l2_px + lamda * l1_px
                
                # Weights for sharpness: REMOVED to match source stablity
                # mask = (yb.abs() > 0.05).to(eps_pred.dtype)
                # weights = 1.0 + 5.0 * mask
                # eps_loss = (eps_loss_px * weights).mean()
                eps_loss = eps_loss_px.mean()
            else:
                raise ValueError(f"Unknown LOSS_OBJECTIVE={LOSS_OBJECTIVE}")

            # --- x0 auxiliary reconstruction (weighted) ---
            if X0_AUX_WEIGHT > 0:
                # sa can be 0.006 at t=1000; clamping helps prevent explosion
                sa = extract(sqrt_alpha_bars, t, yb.shape)
                so = extract(sqrt_one_minus_alpha_bars, t, yb.shape)
                sa_stable = sa.clamp(min=1e-3)
                x0_hat = (y_t - so * eps_pred) / sa_stable
                
                # L1 is better for structural reconstruction
                x0_loss_px = F.l1_loss(x0_hat, yb, reduction='none')
                # x0_loss = (x0_loss_px * weights).mean()
                x0_loss = x0_loss_px.mean()
                loss = eps_loss + X0_AUX_WEIGHT * x0_loss
                
                # --- Gradient-matching loss (soft regularizer) ---
                # Only apply when t < GRAD_T_MAX (x0_hat is cleaner at low timesteps)
                if USE_GRAD_LOSS and GRAD_LOSS_WEIGHT > 0:
                    # Create mask for samples with t < threshold
                    t_mask = (t < GRAD_T_MAX).float().view(-1, 1, 1, 1)
                    if t_mask.sum() > 0:  # Only compute if any samples qualify
                        grad_pred_x, grad_pred_y = compute_spatial_gradients(x0_hat, mode=GRAD_MODE)
                        grad_gt_x, grad_gt_y = compute_spatial_gradients(yb, mode=GRAD_MODE)
                        
                        # Masked L1 loss (only for low-noise samples)
                        grad_loss_x = (torch.abs(grad_pred_x - grad_gt_x) * t_mask).sum() / (t_mask.sum() * grad_pred_x[0].numel() + 1e-8)
                        grad_loss_y = (torch.abs(grad_pred_y - grad_gt_y) * t_mask).sum() / (t_mask.sum() * grad_pred_y[0].numel() + 1e-8)
                        grad_loss = grad_loss_x + grad_loss_y
                        loss = loss + GRAD_LOSS_WEIGHT * grad_loss
            else:
                loss = eps_loss

        if train:
            if not torch.isfinite(loss):
                print(f"!!! WARNING: NaN/Inf loss ({loss.item()}) detected! Skipping update. !!!")
                opt.zero_grad(set_to_none=True)
                continue

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            # Match original source setting for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) 
            scaler.step(opt)
            scaler.update()
            ema.update(model)

        total += float(loss.item()) * int(B)
        count += int(B)

    return total / max(1, count)

# -------------------------
# Training Loop
# -------------------------
def save_loss_plot(train_losses, val_losses, out_path):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(out_path)
    plt.close()

# -------------------------
# DDIM sampler (eta=0) with EMA (correct drop-in) 
# -------------------------
@torch.inference_mode()
def sample_ddim(
    x_cond,
    steps=200,
    prefer_cuda=True,
    max_batch=1,
    use_ema=True,
    clamp_x0=None,  # e.g. (-5, 5) in normalized space; usually None
):
    steps = int(max(2, min(int(steps), T)))

    if x_cond.dim() == 3:
        x_cond = x_cond.unsqueeze(0)
    if x_cond.shape[0] > max_batch:
        x_cond = x_cond[:max_batch]

    device_order = []
    if prefer_cuda and torch.cuda.is_available():
        device_order.append(torch.device("cuda"))
    device_order.append(torch.device("cpu"))

    orig_device = next(model.parameters()).device

    def _cleanup_cuda(force=False):
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except: pass
            if force:
                torch.cuda.empty_cache()

    def make_ts(dev):
        ts = torch.linspace(T - 1, 0, steps, device=dev).round().long()
        ts = torch.unique_consecutive(ts)
        if ts.numel() < 2:
            ts = torch.tensor([T - 1, 0], device=dev, dtype=torch.long)
        return ts

    for dev in device_order:
        applied = False
        try:
            # Only move if necessary
            if dev != orig_device:
                model.to(dev)

            if use_ema:
                ema.to(dev)  # Only moves if not on dev
                ema.apply_to(model)
                applied = True

            model.eval()

            xm, xs = x_mean.to(dev), x_std.to(dev)
            ym, ys = y_mean.to(dev), y_std.to(dev)
            ab = alpha_bars.to(dev, dtype=torch.float32)

            x = (x_cond.to(dev) - xm) / xs
            B = x.shape[0]
            ts = make_ts(dev)

            y_dtype = torch.float16 if dev.type == "cuda" else torch.float32
            y = torch.randn((B, 1, IMG_H, IMG_W), device=dev, dtype=y_dtype)

            amp_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(dev.type == "cuda"))
                if dev.type == "cuda" else nullcontext()
            )

            with amp_ctx:
                pbar_sampling = tqdm(range(ts.numel()), desc=f"Sampling ({dev})", leave=False)
                for i in pbar_sampling:
                    t = ts[i].expand(B)
                    a_t = ab.gather(0, t).view(B, 1, 1, 1)
                    sqrt_a_t = torch.sqrt(a_t).to(dtype=y.dtype)
                    sqrt_oma_t = torch.sqrt(1.0 - a_t).to(dtype=y.dtype)

                    eps = model(x, y, t)
                    x0_hat = (y - sqrt_oma_t * eps) / sqrt_a_t

                    if clamp_x0 is not None:
                        lo, hi = clamp_x0
                        x0_hat = x0_hat.clamp(lo, hi)

                    if i == ts.numel() - 1:
                        y = x0_hat
                        break

                    t_next = ts[i + 1].expand(B)
                    a_next = ab.gather(0, t_next).view(B, 1, 1, 1)
                    sqrt_a_next = torch.sqrt(a_next).to(dtype=y.dtype)
                    sqrt_oma_next = torch.sqrt(1.0 - a_next).to(dtype=y.dtype)

                    # eta = 0 deterministic DDIM update
                    y = sqrt_a_next * x0_hat + sqrt_oma_next * eps

            y = (y.float() * ys + ym).cpu()

            if applied:
                ema.restore(model)
            
            # Optimization: Only move back if we actually changed devices
            if dev != orig_device:
                model.to(orig_device)
                _cleanup_cuda(force=True) # Only clear cache if we did a heavy move
            else:
                _cleanup_cuda(force=False) # Light sync only

            return y

        except RuntimeError as e:
            if applied:
                try:
                    ema.restore(model)
                except Exception:
                    pass

            if dev.type == "cuda":
                print("CUDA sampling failed; falling back to CPU. Error:", e)
                _cleanup_cuda()
                if dev != orig_device:
                    model.to(orig_device)
                continue

            if dev != orig_device:
                model.to(orig_device)
            raise

    raise RuntimeError("Sampling failed on both CUDA and CPU.")

# -------------------------
# CFD Domain Diagram (Panel 1 replacement)
# -------------------------
def plot_domain_panel(ax, x_input):
    """
    Render a CFD domain diagram: radial mesh, building footprints,
    inlet/outlet boundary arcs with arrows, GAN area marker, and labels.
    """
    H, W = x_input.shape[1], x_input.shape[2]
    cx, cy = W // 2, H // 2
    R = min(H, W) // 2 - 5

    # Wind direction
    sin_val = float(x_input[AUG_CH_DIR_SIN].mean())
    cos_val = float(x_input[AUG_CH_DIR_COS].mean())
    mag = np.sqrt(sin_val**2 + cos_val**2)
    if mag < 1e-6:
        sin_val, cos_val, mag = 1.0, 0.0, 1.0
    wind_dir_deg = np.degrees(np.arctan2(cos_val, sin_val))
    inlet_center_deg = (wind_dir_deg + 180) % 360

    ax.set_facecolor("white")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")

    # Radial mesh
    n_rings = 7
    for i in range(1, n_rings + 1):
        r = R * i / n_rings
        ax.add_patch(plt.Circle((cx, cy), r, fill=False, edgecolor="#d0d0d0",
                                linewidth=0.4, zorder=1))
    for angle_deg in range(0, 360, 30):
        rad = np.radians(angle_deg)
        ax.plot([cx, cx + R * np.cos(rad)], [cy, cy + R * np.sin(rad)],
                color="#d0d0d0", linewidth=0.4, zorder=1)

    # Building footprints
    bldg_mask = x_input[4] > 0
    if np.any(bldg_mask):
        bldg_rgba = np.zeros((H, W, 4), dtype=np.float32)
        bldg_rgba[bldg_mask] = [0.15, 0.15, 0.15, 1.0]
        ax.imshow(bldg_rgba, origin="lower", extent=[0, W, 0, H], zorder=2)

    # GAN Area circle
    if np.any(bldg_mask):
        ys, xs = np.where(bldg_mask)
        max_dist = np.sqrt(((xs - cx)**2 + (ys - cy)**2).max())
        gan_r = np.clip(max_dist * 1.15, R * 0.35, R * 0.85)
    else:
        gan_r = R * 0.6
    ax.add_patch(plt.Circle((cx, cy), gan_r, fill=False, edgecolor="goldenrod",
                             linestyle="--", linewidth=1.5, zorder=3))

    # Inlet/outlet arcs
    ax.add_patch(mpatches.Arc((cx, cy), R * 2, R * 2, angle=0,
                              theta1=inlet_center_deg - 90, theta2=inlet_center_deg + 90,
                              edgecolor="royalblue", linewidth=2.5, fill=False, zorder=4))
    ax.add_patch(mpatches.Arc((cx, cy), R * 2, R * 2, angle=0,
                              theta1=wind_dir_deg - 90, theta2=wind_dir_deg + 90,
                              edgecolor="red", linewidth=2.5, fill=False, zorder=4))

    # Inlet arrows (all parallel toward center)
    n_arrows = 9
    arrow_len = 28
    mid_angle = np.radians(inlet_center_deg)
    dx_arrow = -arrow_len * np.cos(mid_angle)
    dy_arrow = -arrow_len * np.sin(mid_angle)
    for i in range(n_arrows):
        frac = (i + 0.5) / n_arrows
        angle = np.radians(inlet_center_deg - 90 + frac * 180)
        x_start = cx + R * np.cos(angle)
        y_start = cy + R * np.sin(angle)
        ax.annotate("", xy=(x_start + dx_arrow, y_start + dy_arrow), xytext=(x_start, y_start),
                     arrowprops=dict(arrowstyle="->", color="royalblue", lw=1.3), zorder=5)

    # Labels
    inlet_rad = np.radians(inlet_center_deg + 45)
    ax.text(cx + (R + 22) * np.cos(inlet_rad), cy + (R + 22) * np.sin(inlet_rad),
            "inlet", ha="center", va="center", fontsize=15, color="royalblue", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="none"),
            zorder=6)
    outlet_rad = np.radians(wind_dir_deg + 45)
    ax.text(cx + (R + 22) * np.cos(outlet_rad), cy + (R + 22) * np.sin(outlet_rad),
            "outlet", ha="center", va="center", fontsize=15, color="red", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="none"),
            zorder=6)
    gan_label_rad = np.radians(wind_dir_deg + 90)
    ax.text(cx + gan_r * 0.65 * np.cos(gan_label_rad),
            cy + gan_r * 0.65 * np.sin(gan_label_rad),
            "GAN Area", ha="center", va="center", fontsize=9,
            color="goldenrod", fontstyle="italic", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="none"),
            zorder=6)

    ax.set_title("Domain Setup", pad=36, fontsize=15, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="gray", alpha=0.9))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)


# -------------------------
# Save prediction vs GT (Agg, no GUI) with SHARED scaling
# -------------------------
def save_pred_vs_true(y_pred, y_true, out_path="val_pred_vs_true.png", cmap="gray", interpolation="nearest", x=None):
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    
    # force non-negative physics for visualization
    y_pred = np.maximum(y_pred, 0)

    # Circular domain mask — hide pixels outside the domain
    H, W = y_true.shape
    cy_m, cx_m = H // 2, W // 2
    rad_m = min(H, W) // 2 - 5
    Yc_m, Xc_m = np.ogrid[:H, :W]
    outside = np.sqrt((Xc_m - cx_m)**2 + (Yc_m - cy_m)**2) >= rad_m

    y_true_vis = np.ma.masked_where(outside, y_true)
    y_pred_vis = np.ma.masked_where(outside, y_pred)

    # Shared limits based on Ground Truth (inside domain only)
    vmin = 0.0
    vmax = 2.0

    # Check if we have input X for the 4-section plot
    if x is not None:
        # 4-section layout: Input, GT, Pred, Diff
        fig = plt.figure(figsize=(24, 6))
        grid_size = (1, 4)
        has_input = True
        
        # Parse Input (assuming x is tensor or numpy, (C, H, W))
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
            
        # Hardcoded channel for building mask (consistent with inference script)
        CH_BMI = 2 
        mask = x[CH_BMI]
        
        # Wind Vector calculation (Matched to visualize_inference.py)
        sin_val = x[AUG_CH_DIR_SIN].mean()
        cos_val = x[AUG_CH_DIR_COS].mean()
        wind_angle = np.degrees(np.arctan2(sin_val, cos_val))
        wind_angle = (wind_angle + 360) % 360
    else:
        # Fallback 3-section layout
        fig = plt.figure(figsize=(15, 5))
        grid_size = (1, 3)
        has_input = False

    # --- Plotting ---

    # 1. Domain Setup (if available)
    plot_idx = 1
    if has_input:
        ax0 = plt.subplot(grid_size[0], grid_size[1], plot_idx)
        plot_idx += 1
        plot_domain_panel(ax0, x)

    # 2. Ground Truth
    ax1 = plt.subplot(grid_size[0], grid_size[1], plot_idx)
    plot_idx += 1
    ax1.set_title("Ground Truth", pad=36, fontsize=15, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="gray", alpha=0.9))
    im1 = ax1.imshow(y_true_vis, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation, origin='lower')
    if has_input:
        ax1.contour(mask, levels=[0.5], colors='white', linewidths=0.5, origin='lower')
    ax1.set_xticks([])
    ax1.set_yticks([])
    for spine in ax1.spines.values(): spine.set_visible(False)

    # 3. Prediction
    ax2 = plt.subplot(grid_size[0], grid_size[1], plot_idx)
    plot_idx += 1
    ax2.set_title("Prediction", pad=36, fontsize=15, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="gray", alpha=0.9))
    im2 = ax2.imshow(y_pred_vis, cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation, origin='lower')
    if has_input:
        ax2.contour(mask, levels=[0.5], colors='white', linewidths=0.5, origin='lower')
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values(): spine.set_visible(False)

    # 4. Difference with Quantitative Metrics
    ax3 = plt.subplot(grid_size[0], grid_size[1], plot_idx)
    plot_idx += 1
    diff = y_pred - y_true

    # Reuse circular domain mask from above
    domain_mask = ~outside

    # --- Compute Metrics (within circular domain only) ---
    diff_masked = diff[domain_mask]
    abs_diff_masked = np.abs(diff_masked)

    mae = np.mean(abs_diff_masked)
    rmse = np.sqrt(np.mean(diff_masked**2))
    max_err = np.max(abs_diff_masked)

    gt_masked = y_true[domain_mask]
    gt_abs = np.abs(gt_masked)
    valid_for_mape = gt_abs > 0.1
    if np.sum(valid_for_mape) > 0:
        mape = np.mean(abs_diff_masked[valid_for_mape] / gt_abs[valid_for_mape]) * 100.0
    else:
        mape = 0.0

    try:
        from skimage.metrics import structural_similarity as ssim_func
        y_true_ssim = np.nan_to_num(y_true, nan=0.0)
        y_pred_ssim = np.nan_to_num(y_pred, nan=0.0)
        data_range = max(np.max(y_true_ssim), np.max(y_pred_ssim)) - min(np.min(y_true_ssim), np.min(y_pred_ssim))
        ssim_val = ssim_func(y_true_ssim, y_pred_ssim, data_range=data_range) if data_range > 1e-8 else -1.0
    except ImportError:
        ssim_val = -1.0

    def compute_gradient_correlation(pred, true):
        pred_dx = np.diff(pred, axis=1, prepend=pred[:, :1])
        pred_dy = np.diff(pred, axis=0, prepend=pred[:1, :])
        true_dx = np.diff(true, axis=1, prepend=true[:, :1])
        true_dy = np.diff(true, axis=0, prepend=true[:1, :])
        pred_grad = np.concatenate([pred_dx.flatten(), pred_dy.flatten()])
        true_grad = np.concatenate([true_dx.flatten(), true_dy.flatten()])
        mask_flat = np.concatenate([domain_mask.flatten(), domain_mask.flatten()])
        pred_grad = pred_grad[mask_flat]
        true_grad = true_grad[mask_flat]
        if np.std(pred_grad) < 1e-8 or np.std(true_grad) < 1e-8:
            return 0.0
        return np.corrcoef(pred_grad, true_grad)[0, 1]

    grad_corr = compute_gradient_correlation(y_pred, y_true)

    ss_res = np.sum(diff_masked ** 2)
    ss_tot = np.sum((gt_masked - np.mean(gt_masked)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    diff_vis = np.ma.masked_where(outside, diff)
    im3 = ax3.imshow(diff_vis, cmap='RdBu', vmin=-2.0, vmax=2.0, origin='lower')
    ax3.set_title("Diff (Pred - GT)", pad=36, fontsize=15, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="gray", alpha=0.9))

    ssim_str = f"{ssim_val:.3f}" if ssim_val >= 0 else "N/A"
    metrics_line1 = f"MAE:{mae:.3f} | RMSE:{rmse:.3f} | MAPE:{mape:.1f}%"
    metrics_line2 = f"SSIM:{ssim_str} | GradCorr:{grad_corr:.3f} | R\u00b2:{r2:.3f}"

    ax3.set_xticks([])
    ax3.set_yticks([])
    for spine in ax3.spines.values(): spine.set_visible(False)

    plt.tight_layout(pad=1.5)
    fig.canvas.draw()

    # Align all titles at the same y-position in figure coordinates
    all_axes = [ax0, ax1, ax2, ax3] if has_input else [ax1, ax2, ax3]
    fig_ys = []
    for a in all_axes:
        tx, ty = a.title.get_position()
        fig_y = a.transAxes.transform((tx, ty))[1]
        fig_ys.append(fig_y)
    target_fig_y = max(fig_ys)
    for a in all_axes:
        tx, _ = a.title.get_position()
        new_axes_y = a.transAxes.inverted().transform((0, target_fig_y))[1]
        a.title.set_position((tx, new_axes_y))

    fig.canvas.draw()

    # Horizontal colorbars — same width, same y-position
    bb1 = ax1.get_position()
    bb2 = ax2.get_position()
    bb3 = ax3.get_position()
    cb_w = bb3.width
    cb_y = min(bb1.y0, bb3.y0) - 0.12

    shared_center = (bb1.x0 + bb2.x1) / 2
    cax_shared = fig.add_axes([shared_center - cb_w / 2, cb_y, cb_w, 0.025])
    fig.colorbar(im1, cax=cax_shared, orientation="horizontal")

    diff_center = bb3.x0 + bb3.width / 2
    cax_diff = fig.add_axes([diff_center - cb_w / 2, cb_y, cb_w, 0.025])
    fig.colorbar(im3, cax=cax_diff, orientation="horizontal")

    # Metrics text below diff colorbar
    metrics_y = cb_y - 0.10
    fig.text(diff_center, metrics_y, f"{metrics_line1}\n{metrics_line2}",
             ha="center", va="top", fontsize=10, family="monospace",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    plt.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close("all")
    print("Saved:", out_path)
    print(f"Shared scale vmin={vmin} vmax={vmax} | MAE={mae:.4f} RMSE={rmse:.4f} MAPE={mape:.1f}% SSIM={ssim_str} GradCorr={grad_corr:.3f} R²={r2:.4f}")
    
    # Return metrics for external logging
    return {"mae": mae, "rmse": rmse, "mape": mape, "max_err": max_err, "ssim": ssim_val, "grad_corr": grad_corr, "r2": r2}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="WindTransformer Training Script")
    parser.add_argument("--out_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--augment", action="store_true", help="Enable online dataset augmentation (flips)")
    parser.add_argument("--clip", type=float, default=0.5, help="Gradient clipping value (baseline=0.5, safety=0.1)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (Scenario D used 5e-4)")
    parser.add_argument("--aux", type=float, default=0.01, help="Auxiliary loss weight (Scenario E uses 0.01)")
    parser.add_argument("--grad", type=float, default=0.0, help="Gradient-matching loss weight (0 to disable, try 0.01-0.1)")
    parser.add_argument("--grad_mode", type=str, default="sobel", choices=["simple", "sobel"], help="Gradient kernel: 'simple' ([-1,0,1]/2) or 'sobel' (3x3)")
    parser.add_argument("--grad_t_max", type=int, default=500, help="Only apply grad loss when t < this threshold (helps at low noise)")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--patch", type=int, default=9, help="Patch size (must divide 504 evenly)")
    args = parser.parse_args()

    # Validate patch size
    if 504 % args.patch != 0:
        raise ValueError(f"Patch size {args.patch} does not divide 504 evenly. Valid options: 1,2,3,4,6,7,8,9,12,14,18,21,24,28,36,42,56,63,72,84,126,168,252,504")

    # Apply configuration overrides
    LR = args.lr
    X0_AUX_WEIGHT = args.aux
    GRAD_LOSS_WEIGHT = args.grad
    GRAD_MODE = args.grad_mode
    GRAD_T_MAX = args.grad_t_max
    EPOCHS = args.epochs
    PATCH = args.patch
    # Recalculate window parameters BASED ON THE NEW PATCH SIZE
    recalculate_window_params(PATCH, silent=False)
    update_dirs(args.out_dir)

    # Update CONFIG dictionary with final values after CLI overrides
    CONFIG.update({
        "LR": LR,
        "X0_AUX_WEIGHT": X0_AUX_WEIGHT,
        "GRAD_LOSS_WEIGHT": GRAD_LOSS_WEIGHT,
        "GRAD_MODE": GRAD_MODE,
        "GRAD_T_MAX": GRAD_T_MAX,
        "EPOCHS": EPOCHS,
        "PATCH": PATCH,
        "OUT_DIR": OUT_DIR,
        "AUGMENT": args.augment,
        "CLIP": args.clip,
        "SLURM_JOB_ID": os.environ.get("SLURM_JOB_ID", "local"),
        "SLURM_NODE": os.environ.get("SLURM_JOB_NODELIST", "local"),
        "WINDOW_SIZE": WINDOW_SIZE
    })

    # Save configuration to a JSON file for record-keeping (AS EARLY AS POSSIBLE)
    config_save_path = os.path.join(OUT_DIR, "training_config.json")
    with open(config_save_path, "w") as f:
        json.dump(CONFIG, f, indent=4)
    
    print(f"Configuration saved to: {config_save_path}")
    print(f"Augmentation enabled: {args.augment}")
    print(f"Assigned LR: {LR} | Aux: {X0_AUX_WEIGHT} | Grad: {GRAD_LOSS_WEIGHT} (mode={GRAD_MODE}, t_max={GRAD_T_MAX}) | Epochs: {EPOCHS} | Patch: {PATCH}")

    # --- Build model / optim / EMA (Moved inside main for CLI safety) ---
    model = PatchTransformerDenoiser(
        in_ch=X_CH + Y_CH,
        patch=PATCH,
        emb=EMB,
        depth=DEPTH,
        heads=HEADS,
        mlp_ratio=MLP_RATIO,
        dropout=DROPOUT,
        img_hw=(IMG_H, IMG_W),
        attn_mode=ATTN_MODE,
        window_size=WINDOW_SIZE,
        shift_windows=SHIFT_WINDOWS,
        use_refiner=USE_REFINER,
        refiner_use_yt=REFINER_USE_YT,
        refiner_use_x=REFINER_USE_X,
        refiner_x_proj_ch=REFINER_X_PROJ_CH,
    ).to(device)

    if USE_TORCH_COMPILE:
        try:
            print("!!! Model compiling with torch.compile (this takes 2-5 minutes on first run) !!!")
            model = torch.compile(model, mode="default")
            print("torch.compile enabled")
        except Exception as e:
            print("torch.compile failed, continuing without it. Error:", e)

    ema = EMA(model, decay=EMA_DECAY)
    ema.to(device)

    try:
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, fused=True)
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gammad)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    print("Total parameters:", sum(p.numel() for p in model.parameters()))
    print(f"Tokens: ph={IMG_H//PATCH}, pw={IMG_W//PATCH}, N={(IMG_H//PATCH)*(IMG_W//PATCH)} | "
          f"attn_mode={ATTN_MODE} window={WINDOW_SIZE} shift={SHIFT_WINDOWS}")

    # Initialize data loaders
    try:
        X_mm, Y_mm, train_idx, val_idx = get_memmap_data()
        train_ds = XYMemmapDataset(X_mm, Y_mm, train_idx, x_ch=X_CH, img_hw=(IMG_H, IMG_W), augment=args.augment)
        val_ds   = XYMemmapDataset(X_mm, Y_mm, val_idx,   x_ch=X_CH, img_hw=(IMG_H, IMG_W), augment=False)

        # Initialize stats after dataset is ready
        load_or_compute_stats(train_ds)

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=2 if NUM_WORKERS > 0 else None,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=PERSISTENT_WORKERS,
            prefetch_factor=2 if NUM_WORKERS > 0 else None,
        )
        print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")
    except Exception as e:
        print(f"FATAL ERROR: Could not initialize data/loaders: {e}")
        import sys
        sys.exit(1)

    print("Starting training...")
    best_loss = float('inf')
    train_losses = []
    val_losses = []
    start_epoch = 1

    # Resume from checkpoint if exists
    resume_path = os.path.join(CKPT_DIR, "wind_checkpoint.pth")
    best_path   = os.path.join(CKPT_DIR, "wind_model_best.pth")
    checkpoint = None

    if os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        # Check for corruption
        if checkpoint.get('loss') is not None and (math.isnan(checkpoint['loss']) or checkpoint['loss'] == 0.0):
            print(f"!!! WARNING: Checkpoint '{resume_path}' has NaN/Zero loss. Ignoring it.")
            checkpoint = None
    
    # Fallback to best if valid checkpoint not found yet
    if checkpoint is None and os.path.exists(best_path):
        print(f"Resuming from BEST checkpoint: {best_path}")
        checkpoint = torch.load(best_path, map_location=device)

    if checkpoint is not None:
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint:
                try:
                    opt.load_state_dict(checkpoint['optimizer_state_dict'])
                except Exception as e:
                    print(f"Warning: Could not load optimizer state (skipping): {e}")
            
                    print(f"Warning: Could not load optimizer state (skipping): {e}")
            
            # Resumed optimizer state correctly.
            
            if 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
                ema.load_state_dict(checkpoint['ema_state_dict'])
                print("Resumed EMA state.")
            else:
                print("WARNING: EMA state not found in checkpoint! EMA will start from current model weights (suboptimal).") 
                # Re-init EMA with current model weights if missing
                ema = EMA(model, decay=EMA_DECAY)
                ema.to(device)

            start_epoch = checkpoint.get('epoch', 0) + 1
            best_loss = checkpoint.get('best_loss', float('inf'))
            train_losses = checkpoint.get('train_losses', [])
            val_losses = checkpoint.get('val_losses', [])
            print(f"Resumed from dictionary checkpoint. Start epoch: {start_epoch}")

            # --- ENSURE SCHEDULER RESUMES AT CORRECT EPOCH (Strict Continuity) ---
            # We want the scheduler to pick up exactly where it left off.
            # If start_epoch is 121, it means epochs 1..120 are done.
            # After 120 steps, last_epoch should be 119 in PyTorch internal terms.
            scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gammad, last_epoch=start_epoch - 2)
            print(f"!!! Scheduler synchronized to Epoch {start_epoch-1} (LR: {opt.param_groups[0]['lr']:.2e}) !!!")
        else:
            # Legacy format: The checkpoint is just the state_dict
            model.load_state_dict(checkpoint)
            print("Resumed from legacy (state_dict only) checkpoint. Resetting epoch to 1.")
            start_epoch = 1
            best_loss = float('inf')
            train_losses = []
            val_losses = []

    else:
        print("No valid checkpoint found. Starting fresh.")

        # Generate image immediately on resume
        # [OPTIMIZATION] Skip generation to start training immediately
        # print("Generating validation image on resume...")
        # try:
        #     j_res = np.random.randint(0, len(val_ds))
        #     xr, yr = val_ds[j_res]
        #     yp_t = sample_ddim(xr.unsqueeze(0), steps=SAMPLE_STEPS, prefer_cuda=True, max_batch=1, use_ema=True)
        #     save_pred_vs_true(
        #         yp_t[0, 0].numpy(), yr[0].numpy(),
        #         out_path=os.path.join(IMG_DIR, f"val_resume_epoch_{checkpoint['epoch']:03d}.png"),
        #         cmap="viridis", interpolation="bilinear"
        #     )
        # except Exception as e:
        #     print(f"  --> Failed to generate image on resume: {e}")

    for epoch in range(start_epoch, EPOCHS + 1):
        tr = run_epoch(train_loader, train=True)
        va = run_epoch(val_loader, train=False)
        scheduler.step()
        
        train_losses.append(tr)
        val_losses.append(va)
        
        # Display current LR in log for monitoring
        actual_lr = opt.param_groups[0]['lr']
        print(f"Epoch {epoch:03d} | train loss: {tr:.6f} | val loss: {va:.6f} | lr: {actual_lr:.2e}")

        # Best weights logic
        if va < best_loss:
            best_loss = va
            best_path = os.path.join(CKPT_DIR, "wind_model_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': va,
                'best_loss': best_loss,
                'ema_state_dict': ema.state_dict() if ema else None,
                'config': CONFIG,
            }, best_path)
            print(f"  --> New best loss! Saved {best_path}")


        # 1. Visualization & Loss Plotting (Every 10 epochs)
        if epoch % 10 == 0:
            # Update Plots & CSV
            save_loss_plot(train_losses, val_losses, os.path.join(PLOT_DIR, "loss_curve.png"))
            
            with open(os.path.join(PLOT_DIR, "loss_history.csv"), "w") as f:
                f.write("epoch,train_loss,val_loss\n")
                for i, (t, v) in enumerate(zip(train_losses, val_losses)):
                    f.write(f"{i+1},{t},{v}\n")
            
            # Save Validation Image
            j = np.random.randint(0, len(val_ds))
            x_raw, y_raw = val_ds[j]

            y_pred_t = sample_ddim(
                x_raw.unsqueeze(0),
                steps=PLOT_STEPS,
                prefer_cuda=True,
                max_batch=1,
                use_ema=True,
                clamp_x0=None,
            )

            y_pred = y_pred_t[0, 0].numpy()
            y_true = y_raw[0].numpy()

            metrics = save_pred_vs_true(
                y_pred, y_true,
                out_path=os.path.join(IMG_DIR, f"val_epoch_{epoch:03d}.png"),
                cmap="viridis",
                interpolation="bilinear",
                x=x_raw
            )
            print(f"  --> Saved validation image: val_epoch_{epoch:03d}.png")
            
            # Append metrics to history CSV
            metrics_csv_path = os.path.join(PLOT_DIR, "metrics_history.csv")
            expected_header = "epoch,mae,rmse,mape,max_err,ssim,grad_corr,r2\n"
            write_header = not os.path.exists(metrics_csv_path)
            # If file exists but header is outdated (e.g. missing r2), fix it
            if not write_header:
                with open(metrics_csv_path, "r") as f:
                    first_line = f.readline()
                if "r2" not in first_line:
                    with open(metrics_csv_path, "r") as f:
                        lines = f.readlines()
                    with open(metrics_csv_path, "w") as f:
                        f.write(expected_header)
                        f.writelines(lines[1:])  # skip old header
                    print(f"  Updated metrics_history.csv header to include r2")
            with open(metrics_csv_path, "a") as f:
                if write_header:
                    f.write(expected_header)
                if metrics:
                    f.write(f"{epoch},{metrics['mae']:.6f},{metrics['rmse']:.6f},{metrics['mape']:.2f},{metrics['max_err']:.4f},{metrics['ssim']:.4f},{metrics['grad_corr']:.4f},{metrics['r2']:.4f}\n")

        # 2. Checkpoint Saving (Every 20 epochs)
        if epoch % 20 == 0:
            ckpt_path = os.path.join(CKPT_DIR, "wind_checkpoint.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': va,
                'best_loss': best_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'ema_state_dict': ema.state_dict() if ema else None,
                'config': CONFIG,
            }, ckpt_path)
            print(f"  --> Saved checkpoint: {ckpt_path}")

######################################################
#### WE PULL TESTING SAMPLES FROM VALIDATION DATA ####
######################################################


