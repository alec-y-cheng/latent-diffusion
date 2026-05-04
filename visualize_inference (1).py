import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import matplotlib.patches as patches

# Import from the main script
import sys
sys.path.append(os.getcwd())

# Building Mask Channel (Assumed)
CH_BUILDING = 2 

def load_training_config(checkpoint_path):
    """
    Attempt to load training_config.json from the checkpoint's results folder.
    Returns dict with config values or empty dict if not found.
    """
    # Checkpoint is typically at: results_xxx/checkpoints/wind_model_best.pth
    # Config is at: results_xxx/training_config.json
    ckpt_dir = os.path.dirname(checkpoint_path)  # .../checkpoints
    results_dir = os.path.dirname(ckpt_dir)       # .../results_xxx
    
    config_path = os.path.join(results_dir, "training_config.json")
    
    if os.path.exists(config_path):
        print(f"Found training config: {config_path}")
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    else:
        print(f"No training_config.json found at {config_path}")
        return {}

def visualize_inference(
    checkpoint_path, 
    num_samples=5, 
    output_dir="results/vis_inference",
    steps=200,
    device_id="cuda",
    patch_size_override=None
):
    """Run inference visualization with optional patch size override."""
    
    # Import module (may have been modified by patch_size override)
    import WindTransformer_windowed as WTW
    
    # Load training config from checkpoint folder
    train_config = load_training_config(checkpoint_path)
    
    # Determine patch size: CLI override > training_config.json > current script default
    if patch_size_override is not None:
        patch_size = patch_size_override
        print(f"[OVERRIDE] Using CLI patch_size: {patch_size}")
    elif "PATCH" in train_config:
        patch_size = train_config["PATCH"]
        print(f"[CONFIG] Using PATCH from training_config.json: {patch_size}")
    else:
        patch_size = WTW.PATCH
        print(f"[DEFAULT] Using current script PATCH: {patch_size}")
    
    # Apply patch size if different from current
    if patch_size != WTW.PATCH:
        old_patch = WTW.PATCH
        WTW.PATCH = patch_size
        WTW.recalculate_window_params(patch_size)
        print(f"[OVERRIDE] PATCH: {old_patch} -> {patch_size}")
    
    # Now get the (potentially updated) values
    PATCH = WTW.PATCH
    EMB = WTW.EMB
    DEPTH = WTW.DEPTH
    HEADS = WTW.HEADS
    MLP_RATIO = WTW.MLP_RATIO
    DROPOUT = WTW.DROPOUT
    IMG_H = WTW.IMG_H
    IMG_W = WTW.IMG_W
    X_CH = WTW.X_CH
    Y_CH = WTW.Y_CH
    ATTN_MODE = WTW.ATTN_MODE
    WINDOW_SIZE = WTW.WINDOW_SIZE
    SHIFT_WINDOWS = WTW.SHIFT_WINDOWS
    USE_REFINER = WTW.USE_REFINER
    REFINER_USE_YT = WTW.REFINER_USE_YT
    REFINER_USE_X = WTW.REFINER_USE_X
    REFINER_X_PROJ_CH = WTW.REFINER_X_PROJ_CH
    EMA_DECAY = WTW.EMA_DECAY
    device = WTW.device
    AUG_CH_DIR_SIN = WTW.AUG_CH_DIR_SIN
    AUG_CH_DIR_COS = WTW.AUG_CH_DIR_COS
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model from {checkpoint_path}...")
    print(f"Using PATCH={PATCH}, WINDOW_SIZE={WINDOW_SIZE}")
    
    # Initialize model with current params
    model = WTW.PatchTransformerDenoiser(
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
    

    # Inject model and EMA into global scope for sample_ddim
    WTW.model = model
    ema = WTW.EMA(model, decay=EMA_DECAY)
    ema.to(device)
    WTW.ema = ema
    
    # Load Weights
    ema_loaded = False
    try:
        data = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in data:
            model.load_state_dict(data['model_state_dict'])
            
            # Try to load EMA
            if 'ema_state_dict' in data and data['ema_state_dict'] is not None:
                print("EMA weights found in checkpoint! Loading...")
                ema.load_state_dict(data['ema_state_dict'])
                ema.to(device)
                ema_loaded = True
            else:
                print("WARNING: No EMA weights in checkpoint. Using raw weights (expect noise).")
        else:
            model.load_state_dict(data)
            print("Legacy checkpoint (no EMA). Using raw weights.")
            
        print("Weights loaded.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    # Load Data
    print("Loading data...")
    X_mm, Y_mm, train_idx, val_idx = WTW.get_memmap_data(override_ram=False)
    
    # CRITICAL: Load normalization statistics (x_mean, x_std, etc.)
    # Without this, sample_ddim uses placeholder values and produces garbage!
    print("Loading normalization statistics...")
    train_ds_for_stats = WTW.XYMemmapDataset(X_mm, Y_mm, train_idx, x_ch=X_CH, img_hw=(IMG_H, IMG_W))
    WTW.load_or_compute_stats(train_ds_for_stats)
    
    val_ds = WTW.XYMemmapDataset(X_mm, Y_mm, val_idx, x_ch=X_CH, img_hw=(IMG_H, IMG_W))
    
    indices = np.random.choice(len(val_ds), num_samples, replace=False)
    print(f"Selected {num_samples} samples: {indices}")

    for i, idx in enumerate(indices):
        x_raw, y_raw = val_ds[idx] # (8, H, W), (1, H, W)
        
        # Get Wind Vector for logging
        sin_val = x_raw[AUG_CH_DIR_SIN].mean().item()
        cos_val = x_raw[AUG_CH_DIR_COS].mean().item()
        wind_angle = np.degrees(np.arctan2(sin_val, cos_val))
        wind_angle = (wind_angle + 360) % 360
        print(f"Sample {idx}: Sin={sin_val:.2f}, Cos={cos_val:.2f} -> Angle={wind_angle:.1f} deg")

        # Run Inference
        model.eval()
        with torch.no_grad():
            y_pred_t = WTW.sample_ddim(
                x_raw.unsqueeze(0),
                steps=steps,
                prefer_cuda=True,
                max_batch=1,
                use_ema=ema_loaded 
            )
        
        y_pred = y_pred_t[0, 0].cpu().numpy()
        y_true = y_raw[0].cpu().numpy()
        x_np = x_raw.cpu().numpy()

        # Use the same plotting function as training for consistent format and metrics
        out_file = os.path.join(output_dir, f"vis_sample_{i:02d}_idx_{idx}.png")
        WTW.save_pred_vs_true(y_pred, y_true, out_path=out_file, cmap="viridis", x=x_np)
        print(f"Saved {out_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="results/vis_inference")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--patch_size", type=int, default=None, 
                        help="Override patch size (auto-detected from training_config.json if not specified)")
    args = parser.parse_args()
    
    visualize_inference(
        args.checkpoint, 
        num_samples=args.num_samples, 
        output_dir=args.output_dir,
        steps=args.steps,
        patch_size_override=args.patch_size
    )
