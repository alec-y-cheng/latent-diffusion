
import argparse
import os
import sys
sys.path.append(os.getcwd()) # Ensure ldm is found
import torch
import numpy as np
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pandas as pd

try:
    from skimage.metrics import structural_similarity as ssim_func
except ImportError:
    ssim_func = None

# --- Visualization Logic (Standardized) ---

def compute_gradient_correlation(pred, true, dmask=None):
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

def plot_domain_panel(ax, x_input, ch_x, ch_y, target_W=None, target_H=None):
    """
    Render a CFD domain diagram: radial mesh, building footprints,
    inlet/outlet boundary arcs with arrows, LDM Model boundary, and labels.
    """
    import matplotlib.patches as mpatches
    orig_H, orig_W = x_input.shape[1], x_input.shape[2]
    H = target_H if target_H is not None else orig_H
    W = target_W if target_W is not None else orig_W
    
    cx, cy = W // 2, H // 2
    R = min(H, W) // 2 - 5

    # Wind direction
    sin_val = float(ch_x)
    cos_val = float(ch_y)
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

    # Building footprints (Assuming SDF is at channel 0, negative is inside)
    bldg_mask = x_input[0] <= 0
    if np.any(bldg_mask):
        bldg_rgba = np.zeros((orig_H, orig_W, 4), dtype=np.float32)
        bldg_rgba[bldg_mask] = [0.15, 0.15, 0.15, 1.0]
        ax.imshow(bldg_rgba, origin="lower", extent=[0, W, 0, H], zorder=2)

    # LDM Model bound circle
    if np.any(bldg_mask):
        ys, xs = np.where(bldg_mask)
        scale_x = W / orig_W
        scale_y = H / orig_H
        xs_scaled = xs * scale_x
        ys_scaled = ys * scale_y
        max_dist = np.sqrt(((xs_scaled - cx)**2 + (ys_scaled - cy)**2).max())
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

    # Inlet arrows
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
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="none"), zorder=6)
            
    outlet_rad = np.radians(wind_dir_deg + 45)
    ax.text(cx + (R + 22) * np.cos(outlet_rad), cy + (R + 22) * np.sin(outlet_rad),
            "outlet", ha="center", va="center", fontsize=15, color="red", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="none"), zorder=6)
            
    gan_label_rad = np.radians(wind_dir_deg + 90)
    ax.text(cx + gan_r * 0.65 * np.cos(gan_label_rad),
            cy + gan_r * 0.65 * np.sin(gan_label_rad),
            "LDM Model", ha="center", va="center", fontsize=9,
            color="goldenrod", fontstyle="italic", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="none"), zorder=6)

    ax.set_title("Domain Setup", pad=36, fontsize=15, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="gray", alpha=0.9))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

def save_standardized_plot(y_true, y_pred, save_path, input_cond=None):
    if y_true.ndim > 2: y_true = y_true.squeeze()
    if y_pred.ndim > 2: y_pred = y_pred.squeeze()
    
    H, W = y_true.shape
    diff = y_pred - y_true
    
    # Shared limits (Hardcoded per WindTransformer_windowed reference)
    vmin = 0.0
    vmax = 2.0
    
    # --- Domain Masking (Circular) ---
    center_y, center_x = H // 2, W // 2
    radius = min(H, W) // 2 - 5
    Y_coords, X_coords = np.ogrid[:H, :W]
    dist = np.sqrt((X_coords - center_x)**2 + (Y_coords - center_y)**2)
    domain_mask = dist < radius
    outside = ~domain_mask
    
    # Force non-negative physics for visualization
    y_pred = np.maximum(y_pred, 0)
    
    y_true_vis = np.ma.masked_where(outside, y_true)
    y_pred_vis = np.ma.masked_where(outside, y_pred)
    diff_vis = np.ma.masked_where(outside, diff)
    
    # --- Metrics ---
    diff_masked = diff[domain_mask]
    abs_diff_masked = np.abs(diff_masked)
    
    mae = np.mean(abs_diff_masked)
    rmse = np.sqrt(np.mean(diff_masked**2))
    
    gt_masked = y_true[domain_mask]
    gt_abs = np.abs(gt_masked)
    valid_for_mape = gt_abs > 0.1
    if np.any(valid_for_mape):
        mape = np.mean(abs_diff_masked[valid_for_mape] / gt_abs[valid_for_mape]) * 100.0
    else:
        mape = 0.0

    if ssim_func:
        # Range is now 0 to 2
        ssim_val = ssim_func(y_true, y_pred, data_range=2.0)
    else:
        ssim_val = -1.0
        
    grad_corr = compute_gradient_correlation(y_pred, y_true, domain_mask)
    
    # R² (coefficient of determination) within circular domain
    ss_res = np.sum(diff_masked ** 2)
    ss_tot = np.sum((gt_masked - np.mean(gt_masked)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    
    # --- Plotting ---
    fig = plt.figure(figsize=(24, 6))
    grid_size = (1, 4)
    
    # 1. Domain Setup
    ax0 = plt.subplot(grid_size[0], grid_size[1], 1)
    if input_cond is not None and input_cond.shape[0] >= 8:
        if input_cond.shape[0] == 15:
            ch_x = input_cond[13].mean()
            ch_y = input_cond[14].mean()
        else:
            ch_x = input_cond[6].mean()
            ch_y = input_cond[7].mean()
        plot_domain_panel(ax0, input_cond, ch_x, ch_y, target_W=W, target_H=H)
    elif input_cond is not None and input_cond.shape[0] > 0:
        ax0.imshow(input_cond[0], cmap='gray', origin='lower')
        ax0.set_title("Input (Mask)")
    else:
        ax0.text(0.5, 0.5, "No Vis", ha='center')
        ax0.set_title("Input")

    # 2. Ground Truth
    ax1 = plt.subplot(grid_size[0], grid_size[1], 2)
    ax1.set_title("Ground Truth", pad=36, fontsize=15, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="gray", alpha=0.9))
    im1 = ax1.imshow(y_true_vis, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest', origin='lower')
    if input_cond is not None and input_cond.shape[0] > 0:
        # Solid black buildings per request
        ax1.contour(input_cond[0] <= 0, levels=[0.5], colors='white', linewidths=0.5, origin='lower', extent=[0, W, 0, H])
    ax1.set_xticks([]); ax1.set_yticks([])
    for spine in ax1.spines.values(): spine.set_visible(False)

    # 3. Prediction
    ax2 = plt.subplot(grid_size[0], grid_size[1], 3)
    ax2.set_title("Prediction", pad=36, fontsize=15, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="gray", alpha=0.9))
    im2 = ax2.imshow(y_pred_vis, cmap='viridis', vmin=vmin, vmax=vmax, interpolation='nearest', origin='lower')
    if input_cond is not None and input_cond.shape[0] > 0:
        # Solid black buildings per request
        ax2.contour(input_cond[0] <= 0, levels=[0.5], colors='white', linewidths=0.5, origin='lower', extent=[0, W, 0, H])
    ax2.set_xticks([]); ax2.set_yticks([])
    for spine in ax2.spines.values(): spine.set_visible(False)

    # 4. Difference
    ax3 = plt.subplot(grid_size[0], grid_size[1], 4)
    im3 = ax3.imshow(diff_vis, cmap='RdBu', vmin=-2.0, vmax=2.0, origin='lower', interpolation='nearest')
    ax3.set_title("Diff (Pred - GT)", pad=36, fontsize=15, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="gray", alpha=0.9))
    ax3.set_xticks([]); ax3.set_yticks([])
    for spine in ax3.spines.values(): spine.set_visible(False)
    # --- Precise Legend Placement (Matching Reference) ---
    plt.tight_layout(pad=1.5)
    fig.canvas.draw()
    
    # Get positions after tight_layout
    bb1 = ax1.get_position()
    bb2 = ax2.get_position()
    bb3 = ax3.get_position()
    cb_w = bb3.width
    cb_y = min(bb1.y0, bb3.y0) - 0.12 # Leave space under plots
    
    # Shared horizontal colorbar for GT/Pred
    shared_center = (bb1.x0 + bb2.x1) / 2
    cax_shared = fig.add_axes([shared_center - cb_w / 2, cb_y, cb_w, 0.025])
    fig.colorbar(im1, cax=cax_shared, orientation="horizontal")
    
    # Horizontal colorbar for Diff
    diff_center = bb3.x0 + bb3.width / 2
    cax_diff = fig.add_axes([diff_center - cb_w / 2, cb_y, cb_w, 0.025])
    fig.colorbar(im3, cax=cax_diff, orientation="horizontal")
    
    # Metrics display below colorbars
    ssim_str = f"{ssim_val:.3f}" if ssim_val >= 0 else "N/A"
    metrics_line1 = f"MAE:{mae:.3f} | RMSE:{rmse:.3f} | MAPE:{mape:.1f}%"
    metrics_line2 = f"SSIM:{ssim_str} | GradCorr:{grad_corr:.3f} | R²:{r2:.3f}"
    
    fig.text(diff_center, cb_y - 0.10, f"{metrics_line1}\n{metrics_line2}",
             ha="center", va="top", fontsize=10, family="monospace",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    # Save with enough bottom padding for the new legends
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close()
    
    return {'mae': mae, 'rmse': rmse, 'r2': r2, 'ssim': ssim_val, 'grad_corr': grad_corr, 'mape': mape}

# --- Main Logic ---

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--outdir", type=str, default="test_results", help="Output directory")
    parser.add_argument("--steps", type=int, default=50, help="DDIM Steps")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of random samples to test")
    parser.add_argument("--data_path", type=str, default=None, help="Override dataset path from config")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    
    # 1. Load Config & Model
    config = OmegaConf.load(args.config)
    
    print("Loading Model...")
    try:
        model = load_model_from_config(config, args.ckpt)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
        
    sampler = DDIMSampler(model)
    
    # 2. Load Validation Dataset
    print("Loading Validation Dataset...")
    dataset_conf = config.data.params.validation
    
    # Override data path if provided
    if args.data_path:
        print(f"Overriding dataset path with: {args.data_path}")
        dataset_conf.params.data_path = args.data_path
    
    dataset = instantiate_from_config(dataset_conf)
    
    # Force validation set augmentation to OFF to ensure GT and Predictions align
    if hasattr(dataset, 'augment'):
        dataset.augment = False
    
    # 3. Select Random Indices
    total_len = len(dataset)
    if total_len == 0:
        print(f"CRITICAL ERROR: Dataset is empty! loaded from {dataset_conf.params.data_path}")
        print("Please check if the file exists or provide --data_path argument.")
        return

    indices = np.random.choice(total_len, args.num_samples, replace=False)
    print(f"Selected indices: {indices}")
    
    # 4. Inference Loop
    all_metrics = []
    
    for i, idx in enumerate(tqdm(indices)):
        item = dataset[idx] # Returns dict {'image': ..., 'cond': ...}
        
        # Prepare Batch (Size 1)
        # Note: 'image' is usually (H, W, C) or (C, H, W) depending on transforms
        x_raw = item['image']
        c_raw = item['cond']
        
        # Convert to Tensor (1, C, H, W)
        # Assuming dataset returns Tensors:
        if isinstance(x_raw, torch.Tensor):
            x_gt = x_raw.unsqueeze(0).cuda()
            cond = c_raw.unsqueeze(0).cuda()
        else:
            # If numpy
            x_gt = torch.from_numpy(x_raw).unsqueeze(0).cuda()
            cond = torch.from_numpy(c_raw).unsqueeze(0).cuda()
        
        # Permute if channels last
        if x_gt.ndim == 4 and x_gt.shape[-1] < x_gt.shape[1]: 
             x_gt = x_gt.permute(0, 3, 1, 2)
        if cond.ndim == 4 and cond.shape[-1] < cond.shape[1]: 
             cond = cond.permute(0, 3, 1, 2)
             
        # Sample
        shape = (model.channels, model.image_size, model.image_size)
        
        t0 = time.time()
        with torch.no_grad():
            samples_ddim, _ = sampler.sample(S=args.steps,
                                             conditioning=cond,
                                             batch_size=1,
                                             shape=shape,
                                             verbose=False)
            x_samples = model.decode_first_stage(samples_ddim)
        t1 = time.time()
        inference_time = t1 - t0
            
        # Post-process (Single Sample)
        pred_np = x_samples.cpu().numpy()[0, 0] # (H, W) or (C, H, W)? Assuming single channel target
        gt_np = x_gt.cpu().numpy()[0, 0]
        
        # Un-normalize from LDM [-1, 1] range to [0, 2] per WindTransformer reference
        pred_np = pred_np + 1.0
        gt_np = gt_np + 1.0
        
        # Visualization Input (Conditioning)
        save_path = os.path.join(args.outdir, f"test_sample_{i:03d}_idx_{idx}.png")
        
        # Now we completely defer the plot to the function which also computes metrics
        plot_metrics = save_standardized_plot(gt_np, pred_np, save_path, input_cond=cond.cpu().numpy()[0])
        
        # Collect metrics for aggregation
        metrics = {
            "mae": plot_metrics['mae'],
            "rmse": plot_metrics['rmse'],
            "r2": plot_metrics['r2'],
            "mape": plot_metrics['mape'],
            "ssim": plot_metrics['ssim'],
            "grad_corr": plot_metrics['grad_corr'],
            "inference_time": inference_time
        }
        all_metrics.append(metrics)
        
    print(f"Done. Results saved to {args.outdir}")
    
    # 5. Save Aggregate Metrics
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        summary = df_metrics.agg(['mean', 'std'])
        print("\n--- Aggregate Metrics ---")
        print(summary)
        
        summary_path = os.path.join(args.outdir, "summary_metrics.csv")
        summary.to_csv(summary_path)
        print(f"Summary metrics saved to {summary_path}")
        
        # Save individual metrics too
        all_metrics_path = os.path.join(args.outdir, "all_metrics.csv")
        df_metrics.insert(0, 'idx', indices) # Add index column
        df_metrics.to_csv(all_metrics_path, index=False)

if __name__ == "__main__":
    main()
