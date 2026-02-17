import argparse
import os
import torch
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from skimage.metrics import structural_similarity as ssim_func
except ImportError:
    print("Warning: skimage not found. SSIM will be disabled.")
    ssim_func = None

# --- Metric Helper Functions (Standardized) ---
def compute_gradient_correlation(pred, true, dmask=None):
    # Ensure inputs are 2D
    if pred.ndim > 2: pred = pred.squeeze()
    if true.ndim > 2: true = true.squeeze()
    
    pred_dx = np.diff(pred, axis=1, prepend=pred[:, :1])
    pred_dy = np.diff(pred, axis=0, prepend=pred[:1, :])
    true_dx = np.diff(true, axis=1, prepend=true[:, :1])
    true_dy = np.diff(true, axis=0, prepend=true[:1, :])
    
    pred_grad = np.concatenate([pred_dx.flatten(), pred_dy.flatten()])
    true_grad = np.concatenate([true_dx.flatten(), true_dy.flatten()])
    
    if dmask is not None:
        mask_flat = np.concatenate([dmask.flatten(), dmask.flatten()])
        if len(pred_grad) == len(mask_flat):
            pred_grad = pred_grad[mask_flat]
            true_grad = true_grad[mask_flat]
    
    if np.std(pred_grad) < 1e-6 or np.std(true_grad) < 1e-6:
        return 0.0
        
    return np.corrcoef(pred_grad, true_grad)[0, 1]

def add_border(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1)

def save_standardized_plot(y_true, y_pred, save_path, input_vis=None):
    """
    Generates the standard 4-panel plot: Input (optional), GT, Pred, Diff + Metrics
    y_true, y_pred: 2D numpy arrays (H, W)
    """
    if y_true.ndim > 2: y_true = y_true.squeeze()
    if y_pred.ndim > 2: y_pred = y_pred.squeeze()
    
    H, W = y_true.shape
    diff = y_pred - y_true
    
    # --- Domain Masking (Circular) ---
    center_y, center_x = H // 2, W // 2
    radius = min(H, W) // 2 - 5
    Y_coords, X_coords = np.ogrid[:H, :W]
    dist_from_center = np.sqrt((X_coords - center_x)**2 + (Y_coords - center_y)**2)
    domain_mask = dist_from_center < radius
    
    # --- Compute Metrics ---
    # Filter by domain mask
    diff_masked = diff[domain_mask]
    abs_diff_masked = np.abs(diff_masked)
    
    mae = np.mean(abs_diff_masked)
    rmse = np.sqrt(np.mean(diff_masked**2))
    
    # MAPE
    gt_masked = y_true[domain_mask]
    gt_abs = np.abs(gt_masked)
    valid_for_mape = gt_abs > 0.1 # Avoid div/0
    if np.any(valid_for_mape):
        mape = np.mean(abs_diff_masked[valid_for_mape] / gt_abs[valid_for_mape]) * 100.0
    else:
        mape = 0.0

    # SSIM
    if ssim_func:
        data_range = max(y_true.max(), y_pred.max()) - min(y_true.min(), y_pred.min())
        if data_range == 0: data_range = 1.0
        ssim_val = ssim_func(y_true, y_pred, data_range=data_range)
    else:
        ssim_val = -1.0
        
    # Grad Corr
    grad_corr = compute_gradient_correlation(y_pred, y_true, domain_mask)
    
    # --- Plotting ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # 1. Input (Wind) - Placeholder or actual input if provided
    ax0 = axes[0]
    if input_vis is not None:
        if input_vis.ndim > 2: input_vis = input_vis.squeeze()
        ax0.imshow(input_vis, cmap='viridis', origin='lower')
        ax0.set_title("Input Condition")
    else:
        ax0.text(0.5, 0.5, "No Input Vis", ha='center')
        ax0.set_title("Input Condition")
    
    ax0.set_xlim(0, W)
    ax0.set_ylim(0, H)
    ax0.set_aspect('equal')
    add_border(ax0)
    
    # 2. Ground Truth
    ax1 = axes[1]
    im1 = ax1.imshow(y_true, origin='lower', cmap='viridis')
    ax1.set_title("Ground Truth")
    add_border(ax1)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # 3. Prediction
    ax2 = axes[2]
    im2 = ax2.imshow(y_pred, origin='lower', cmap='viridis')
    ax2.set_title("Prediction")
    add_border(ax2)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # 4. Difference
    ax3 = axes[3]
    # Center colormap around 0
    max_diff = max(abs(diff_masked.min()), abs(diff_masked.max())) if len(diff_masked) > 0 else 1.0
    im3 = ax3.imshow(diff, cmap='RdBu', vmin=-max(1.0, max_diff), vmax=max(1.0, max_diff), origin='lower')
    ax3.set_title("Diff (Pred - GT)")
    
    # Metrics Text
    ssim_str = f"{ssim_val:.3f}" if ssim_val >= 0 else "N/A"
    metrics_line1 = f"MAE:{mae:.3f} | RMSE:{rmse:.3f} | MAPE:{mape:.1f}%"
    metrics_line2 = f"SSIM:{ssim_str} | GradCorr:{grad_corr:.3f}"
    metrics_text = f"{metrics_line1}\n{metrics_line2}"
    ax3.set_xlabel(metrics_text, fontsize=9, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    add_border(ax3)
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    
    return {"mae": mae, "rmse": rmse, "mape": mape, "ssim": ssim_val, "grad_corr": grad_corr}

# --- Core Inference Logic ---

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def csv_to_tensor(csv_path):
    IMG_W, IMG_H = 504, 504
    X_MIN, X_MAX = -485.0, 521.0
    Y_MIN, Y_MAX = -514.5, 491.5
    
    try:
        df = pd.read_csv(csv_path)
        
        # Calculate indices
        x_indices = ((df['X'].values - X_MIN) / (X_MAX - X_MIN) * (IMG_W - 1)).astype(int)
        y_indices = ((df['Y'].values - Y_MIN) / (Y_MAX - Y_MIN) * (IMG_H - 1)).astype(int)
        
        valid = (x_indices >= 0) & (x_indices < IMG_W) & (y_indices >= 0) & (y_indices < IMG_H)
        x_indices = x_indices[valid]
        y_indices = y_indices[valid]
        df = df[valid]
        
        # We need 8 channels for Conditioning (Geometry etc)
        # Assuming order is similar to training: 
        # ['X', 'Y', 'Z_relative', 'SDF', 'Bldg_height', 'U_over_Uref', 'dir_sin', 'dir_cos']
        # Note: 'U_over_Uref' is usually the target? Wait, in training, conditioning is 8 channels.
        # But `cfd_data.py` uses 8 channels for X (Condition) and 1 channel for Y (Target).
        # We need to fill the CONDITIONING here.
        
        feat_cols = ['X', 'Y', 'Z_relative', 'SDF', 'Bldg_height', 'U_over_Uref', 'dir_sin', 'dir_cos']
        
        cond_tensor = np.zeros((8, IMG_H, IMG_W), dtype=np.float32)
        gt_tensor = np.zeros((IMG_H, IMG_W), dtype=np.float32)

        for k, col in enumerate(feat_cols):
             if col in df.columns:
                 cond_tensor[k, y_indices, x_indices] = df[col].values
        
        # Ground Truth for verification
        if 'mag_U' in df.columns:
             gt_tensor[y_indices, x_indices] = df['mag_U'].values
        elif 'U_over_Uref' in df.columns: # Sometimes target is this?
             # Just use it as placeholder if needed
             pass
             
        # Create mask for valid pixels to handle sparsity if needed
        # For now, return dense
        return cond_tensor, gt_tensor, df

    except Exception as e:
        print(f"Error parsing CSV: {e}")
        return None, None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to model config")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--csv", type=str, required=True, help="Input CSV file")
    parser.add_argument("--outdir", type=str, default="inference_results", help="Output directory")
    parser.add_argument("--steps", type=int, default=50, help="DDIM sampling steps")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # 1. Load Model
    config = OmegaConf.load(args.config)
    
    print("Loading Model...")
    model = load_model_from_config(config, args.ckpt)
    sampler = DDIMSampler(model)
    
    # 2. Process Input
    print(f"Reading CSV: {args.csv}")
    cond_504, gt_504, df = csv_to_tensor(args.csv)
    
    if cond_504 is None:
        print("Failed to read CSV.")
        return

    # Prepare Tensor (1, 8, 504, 504)
    cond_tensor = torch.from_numpy(cond_504).float().unsqueeze(0).cuda()
    
    # Resize to 64x64 for Latent Conditioning
    cond_64 = torch.nn.functional.interpolate(cond_tensor, size=(64, 64), mode='bilinear', align_corners=False)
    
    # 3. Sample
    print(f"Sampling with {args.steps} steps...")
    shape = (4, 64, 64) # Latent shape (4 channels, 64x64)
    
    with torch.no_grad():
        samples_ddim, _ = sampler.sample(S=args.steps,
                                         conditioning=cond_64,
                                         batch_size=1,
                                         shape=shape,
                                         verbose=False)
        
        # Decode
        x_samples = model.decode_first_stage(samples_ddim)
        
        # Output is (1, 1, 504, 504) usually (Autoencoder config resolution=504)
        pred_np = x_samples.cpu().numpy()[0, 0]
        
    # 4. Save Results
    base_name = os.path.splitext(os.path.basename(args.csv))[0]
    out_png = os.path.join(args.outdir, f"{base_name}_pred.png")
    out_csv = os.path.join(args.outdir, f"{base_name}_pred.csv")
    
    # Visualize
    # Use Channel 5 (U_over_Uref) as Input Vis
    input_vis = cond_504[5]
    
    metrics = save_standardized_plot(gt_504, pred_np, out_png, input_vis=input_vis)
    
    print(f"Saved visualization to {out_png}")
    print("Metrics:", metrics)
    
    # Save CSV with prediction
    # Needs to map back to CSV rows. 
    # We have (H, W) prediction.
    # df has X, Y. map indices.
    
    # Re-calculate indices for mapping back
    IMG_W, IMG_H = 504, 504
    X_MIN, X_MAX = -485.0, 521.0
    Y_MIN, Y_MAX = -514.5, 491.5
    
    x_indices = ((df['X'].values - X_MIN) / (X_MAX - X_MIN) * (IMG_W - 1)).astype(int)
    y_indices = ((df['Y'].values - Y_MIN) / (Y_MAX - Y_MIN) * (IMG_H - 1)).astype(int)
    
    # Clamp indices to be safe
    x_indices = np.clip(x_indices, 0, IMG_W - 1)
    y_indices = np.clip(y_indices, 0, IMG_H - 1)
    
    pred_vals = pred_np[y_indices, x_indices]
    df['mag_U_pred'] = pred_vals
    
    df.to_csv(out_csv, index=False)
    print(f"Saved CSV to {out_csv}")

if __name__ == "__main__":
    main()
