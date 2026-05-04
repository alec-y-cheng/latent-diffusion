import os
import sys
import argparse
import glob
import torch
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

# Ensure ldm is found
sys.path.append(os.getcwd())

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

try:
    from skimage.metrics import structural_similarity as ssim_func
except ImportError:
    ssim_func = None

# --- Helpers ---

def fix_data_path(data_path):
    """Fix /tmp/ data paths from saved SLURM configs back to real relative paths.
    During training, datasets get cached to /tmp/data/... on the SLURM node.
    The saved project.yaml remembers that /tmp path, but it doesn't exist
    when running inference outside that specific SLURM job."""
    if data_path and data_path.startswith('/tmp/'):
        # Strip /tmp/ prefix to get relative path like data/augmented_did_h5
        relative = data_path.lstrip('/tmp/')
        # Ensure it starts with 'data/' 
        if not relative.startswith('data/'):
            relative = 'data/' + relative
        if os.path.exists(relative):
            print(f"  [fix_data_path] Remapped SLURM temp path {data_path} -> {relative}")
            return relative
    return data_path

def get_experiment_groups(logs_dir, filter_str=None):
    groups = {}
    if not os.path.exists(logs_dir):
        print(f"Error: Logs directory '{logs_dir}' not found.")
        return {}

    if filter_str:
        filters = [f.strip() for f in filter_str.split(',')]
    else:
        filters = None

    for folder in os.listdir(logs_dir):
        path = os.path.join(logs_dir, folder)
        if not os.path.isdir(path):
            continue

        # Filter Logic
        if "autoencoder" in folder.lower():
            continue
        try:
            parts = folder.split('_')
            if len(parts) < 2: continue
            timestamp_str = parts[0]
            exp_name = "_".join(parts[1:])
            dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S")

            # Apply filter against extracted exp_name with exact match
            # to prevent substring collisions (e.g. physics_hybrid_master vs physics_hybrid_master_did)
            if filters and exp_name not in filters:
                continue

            if exp_name not in groups:
                groups[exp_name] = []
            groups[exp_name].append({'path': path, 'timestamp': dt, 'folder': folder})
        except ValueError:
            continue
    return groups

def get_best_checkpoint(folder):
    ckpt_dir = os.path.join(folder, "checkpoints")
    if not os.path.exists(ckpt_dir): return None
    
    best = os.path.join(ckpt_dir, "best.ckpt")
    if os.path.exists(best): return best
    
    last = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.exists(last): return last
    
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts: return None
    
    # Sort by epoch number manually if possible
    # Expect filenames like "epoch=000055.ckpt"
    def get_epoch(path):
        try:
            name = os.path.basename(path)
            # Find "epoch="
            if "epoch=" in name:
                # Extract number after "epoch=" until next non-digit
                part = name.split("epoch=")[1]
                num = ""
                for c in part:
                    if c.isdigit(): num += c
                    else: break
                return int(num)
            return -1
        except:
            return -1
            
    ckpts.sort(key=get_epoch, reverse=True)
    return ckpts[0]

def get_config_path(folder):
    cfg_dir = os.path.join(folder, "configs")
    if not os.path.exists(cfg_dir): return None
    
    cfg = os.path.join(cfg_dir, "project.yaml")
    if os.path.exists(cfg): return cfg
    
    # Check for anything ending in project.yaml (like 2026-02-18-project.yaml)
    yamls = glob.glob(os.path.join(cfg_dir, "*-project.yaml"))
    if yamls: return yamls[0]
    
    # Fallback to any yaml that isn't lightning
    yamls = glob.glob(os.path.join(cfg_dir, "*.yaml"))
    yamls = [y for y in yamls if "lightning" not in y]
    if yamls: return yamls[0]
    
    return None

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

def plot_domain_panel(ax, x_input, target_W=None, target_H=None):
    """
    Render a CFD domain diagram matching WindTransformer_windowed reference:
    radial mesh, building footprints (ch4>0), inlet/outlet boundary arcs,
    GAN area dashed circle, and labels.
    x_input: numpy array (C, H, W) in original data space (un-normalized).
    """
    import matplotlib.patches as mpatches
    orig_H, orig_W = x_input.shape[1], x_input.shape[2]
    H = target_H if target_H is not None else orig_H
    W = target_W if target_W is not None else orig_W
    cx, cy = W // 2, H // 2
    R = min(H, W) // 2 - 5

    # Wind direction — channels 6 (sin) and 7 (cos) per reference AUG_CH_DIR_SIN/COS
    sin_val = float(x_input[6].mean())
    cos_val = float(x_input[7].mean())
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
    if x_input.shape[0] == 15:
        # DID dataset: SDF is at channel 0, building is where SDF <= 0
        bldg_mask = x_input[0] <= 0
    else:
        # Normal 8-channel dataset: building height is at channel 4
        bldg_mask = np.abs(x_input[4]) > 1e-6

    if np.any(bldg_mask):
        bldg_rgba = np.zeros((orig_H, orig_W, 4), dtype=np.float32)
        bldg_rgba[bldg_mask] = [0.15, 0.15, 0.15, 1.0]
        ax.imshow(bldg_rgba, origin="lower", extent=[0, W, 0, H], zorder=2)
    else:
        if x_input.shape[0] == 15:
            print(f"  [plot_domain_panel] WARNING: No buildings detected in ch0. ch0 range: [{x_input[0].min():.4f}, {x_input[0].max():.4f}]")
        else:
            print(f"  [plot_domain_panel] WARNING: No buildings detected in ch4. ch4 range: [{x_input[4].min():.4f}, {x_input[4].max():.4f}]")

    # GAN / LDM area dashed circle
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

    # Inlet arrows pointing inward
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
            "LDM Area", ha="center", va="center", fontsize=9,
            color="goldenrod", fontstyle="italic", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.15", facecolor="white", alpha=0.8, edgecolor="none"), zorder=6)

    ax.set_title("Domain Setup", pad=36, fontsize=15, fontweight="bold",
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke", edgecolor="gray", alpha=0.9))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

def save_standardized_plot(y_true, y_pred, save_path, x_raw_cond=None, c_name=""):
    if y_true.ndim > 2: y_true = y_true.squeeze()
    if y_pred.ndim > 2: y_pred = y_pred.squeeze()

    H, W = y_true.shape
    diff = y_pred - y_true

    # --- Circular domain mask --- pixels outside become transparent
    center_y, center_x = H // 2, W // 2
    radius = min(H, W) // 2
    Y_coords, X_coords = np.ogrid[:H, :W]
    dist = np.sqrt((X_coords - center_x)**2 + (Y_coords - center_y)**2)
    domain_mask = dist < radius
    outside = ~domain_mask
    
    # --- Semantic masking (Floor vs Roof) ---
    hide = outside
    draw_contour = True
    if x_raw_cond is not None and x_raw_cond.shape[0] >= 5:
        if x_raw_cond.shape[0] == 15:
            bldg_mask = x_raw_cond[0] <= 0
        else:
            bldg_mask = x_raw_cond[4] > 0
            
        if bldg_mask.shape != (H, W):
            # Resize mask to match target image resolution (e.g. 504 -> 512)
            tmp = torch.from_numpy(bldg_mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            bldg_mask = torch.nn.functional.interpolate(tmp, size=(H, W), mode='nearest').squeeze().numpy() > 0.5
            
        open_gnd_mask = ~bldg_mask
        
        if "Floor" in c_name:
            hide = outside | bldg_mask
            draw_contour = True
        elif "Roof" in c_name:
            hide = outside | open_gnd_mask
            draw_contour = False

    # --- Metrics (computed inside valid domain) ---
    valid_mask = ~hide
    diff_masked = diff[valid_mask]
    abs_diff_masked = np.abs(diff_masked)
    mae = np.mean(abs_diff_masked) if len(diff_masked) > 0 else 0.0
    rmse = np.sqrt(np.mean(diff_masked**2)) if len(diff_masked) > 0 else 0.0

    gt_masked = y_true[valid_mask]
    gt_abs = np.abs(gt_masked)
    valid_for_mape = gt_abs > 0.1
    mape = np.mean(abs_diff_masked[valid_for_mape] / gt_abs[valid_for_mape]) * 100.0 if np.any(valid_for_mape) else 0.0

    if ssim_func:
        # Range is now 0 to 2
        ssim_val = ssim_func(y_true, y_pred, data_range=2.0)
    else:
        ssim_val = -1.0

    grad_corr = compute_gradient_correlation(y_pred, y_true, valid_mask)
    
    # R² (coefficient of determination) within valid domain
    ss_res = np.sum(diff_masked ** 2)
    ss_tot = np.sum((gt_masked - np.mean(gt_masked)) ** 2) if len(gt_masked) > 0 else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    # --- Masked arrays: hidden pixels → transparent (not dark) ---
    y_true_vis = np.ma.masked_where(hide, y_true)
    y_pred_vis = np.ma.masked_where(hide, y_pred)
    diff_vis   = np.ma.masked_where(hide, diff)

    # Dynamic scale based on the channel
    vmin = 0.0
    vmax = 2.0 if "Speed" in c_name else 0.5

    # Shared title badge style matching WindTransformer reference
    title_kwargs = dict(pad=36, fontsize=15, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="whitesmoke",
                                  edgecolor="gray", alpha=0.9))

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.patch.set_facecolor("white")   # white background so masked = white, not grey

    # Panel 1: Domain Setup
    ax0 = axes[0]
    if x_raw_cond is not None and x_raw_cond.shape[0] >= 8:
        plot_domain_panel(ax0, x_raw_cond, target_W=W, target_H=H)
    elif x_raw_cond is not None and x_raw_cond.shape[0] > 0:
        ax0.imshow(x_raw_cond[0], cmap='gray', origin='lower')
        ax0.set_title("Domain Setup", **title_kwargs)
    else:
        ax0.text(0.5, 0.5, "No Vis", ha='center')
        ax0.set_title("Domain Setup", **title_kwargs)
    # Spines already hidden inside plot_domain_panel; ensure no box frame
    ax0.set_xticks([]); ax0.set_yticks([])
    for sp in ax0.spines.values(): sp.set_visible(False)

    # Panel 2: Ground Truth — shared scale, circular mask, building contour
    ax1 = axes[1]
    im1 = ax1.imshow(y_true_vis, origin='lower', cmap='viridis',
                     vmin=vmin, vmax=vmax, interpolation='nearest')
    ax1.set_facecolor("white")
    ax1.set_title("Ground Truth", **title_kwargs)
    ax1.set_xticks([]); ax1.set_yticks([])
    for sp in ax1.spines.values(): sp.set_visible(False)
    if draw_contour and x_raw_cond is not None and x_raw_cond.shape[0] >= 4:
        if x_raw_cond.shape[0] == 15:
            # DID dataset: SDF is at channel 0, building is where SDF <= 0
            bldg_vis = x_raw_cond[0] <= 0
        else:
            # Normal 8-channel dataset: building height is at channel 4
            bldg_vis = x_raw_cond[4] > 0
        # Filled white overlay for buildings (solid, full opacity)
        if np.any(bldg_vis):
            # Initialize fully transparent, then set building pixels to opaque white
            bldg_overlay = np.zeros((bldg_vis.shape[0], bldg_vis.shape[1], 4), dtype=np.float32)
            bldg_overlay[bldg_vis] = [1.0, 1.0, 1.0, 1.0]  # white, full opacity
            ax1.imshow(bldg_overlay, origin='lower', extent=[0, W, 0, H],
                       interpolation='nearest', zorder=5)


    # Panel 3: Prediction — same shared scale
    ax2 = axes[2]
    im2 = ax2.imshow(y_pred_vis, origin='lower', cmap='viridis',
                     vmin=vmin, vmax=vmax, interpolation='nearest')
    ax2.set_facecolor("white")
    ax2.set_title("Prediction", **title_kwargs)
    ax2.set_xticks([]); ax2.set_yticks([])
    for sp in ax2.spines.values(): sp.set_visible(False)
    if draw_contour and x_raw_cond is not None and x_raw_cond.shape[0] > 4:
        if np.any(bldg_vis):
            # Same solid white overlay on prediction panel
            ax2.imshow(bldg_overlay, origin='lower', extent=[0, W, 0, H],
                       interpolation='nearest', zorder=5)


    # Panel 4: Difference — clipped to [-2, 2] + metrics
    ax3 = axes[3]
    im3 = ax3.imshow(diff_vis, cmap='RdBu', vmin=-2.0, vmax=2.0,
                     origin='lower', interpolation='nearest')
    ax3.set_facecolor("white")
    ax3.set_title("Diff (Pred - GT)", **title_kwargs)
    ax3.set_xticks([]); ax3.set_yticks([])
    for sp in ax3.spines.values(): sp.set_visible(False)

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
    metrics_line2 = f"SSIM:{ssim_str} | GradCorr:{grad_corr:.3f} | R\u00b2:{r2:.3f}"
    
    fig.text(diff_center, cb_y - 0.10, f"{metrics_line1}\n{metrics_line2}",
             ha="center", va="top", fontsize=10, family="monospace",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    # Save with enough bottom padding for the new legends
    plt.savefig(save_path, dpi=150, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close()
    
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "ssim": ssim_val, "grad_corr": grad_corr}



def load_model_from_config(config, ckpt):
    print(f"Loading model state from {ckpt}...")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd
    
    model = instantiate_from_config(config.model)
    model_keys = set(model.state_dict().keys())
    
    # Smart filtering: Only map prefixes if the target key actually exists in the model
    # but is missing from the checkpoint's raw keys.
    sd_new = {}
    for k in model_keys:
        if k in sd:
            sd_new[k] = sd[k]
        elif f"model_ema.{k}" in sd:
            sd_new[k] = sd[f"model_ema.{k}"]
        elif f"model.{k}" in sd:
            sd_new[k] = sd[f"model.{k}"]
            
    # Include any other keys that were in the checkpoint just in case
    for k, v in sd.items():
        if k not in sd_new and not k.startswith("model_ema.") and not k.startswith("model."):
             sd_new[k] = v
    
    missing, unexpected = model.load_state_dict(sd_new, strict=False)
    if len(missing) > 0:
        print(f"  [Warning] Missing {len(missing)} keys in checkpoint")
    if len(unexpected) > 0:
        print(f"  [Warning] Unexpected {len(unexpected)} keys in checkpoint")
        
    model.cuda()
    model.eval()
    return model

# --- Main Fast Batch Logic ---

def main():
    parser = argparse.ArgumentParser(description="Fast Batch Inference (Load Data Once)")
    parser.add_argument("--logs", type=str, default="logs", help="Path to logs directory")
    parser.add_argument("--outdir_suffix", type=str, default="fast_inference_results", help="Dir name inside log folder")
    parser.add_argument("--default_config", type=str, default="configs/latent-diffusion/cfd_ldm.yaml", help="Config for DATASET loading only")
    parser.add_argument("--data_path", type=str, default=None, help="Override validation data path")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--filter", type=str, default=None, help="Filter experiments by name (partial match)")
    args = parser.parse_args()

    # We will determine length and grab indices from the default config just once
    # to ensure all models are compared on the exact same validation images.
    base_config = OmegaConf.load(args.default_config)
    dataset_conf = base_config.data.params.validation
    if args.data_path:
        dataset_conf.params.data_path = args.data_path
    else:
        dataset_conf.params.data_path = fix_data_path(dataset_conf.params.data_path)
    
    # Quick instantiation just to get the length
    dataset = instantiate_from_config(dataset_conf)
    total_len = len(dataset)
    indices = np.random.choice(total_len, args.num_samples, replace=False)
    print(f"Selected unified indices for all models: {indices}")

    # Pre-load data items into CPU memory (if feasible) or just indices
    # To be safe and fast, let's keep it as indices access.

    # 2. Iterate Models
    groups = get_experiment_groups(args.logs, args.filter)
    all_summaries = []

    for exp_name, runs in groups.items():
        runs.sort(key=lambda x: x['timestamp'], reverse=True)
        
        valid_model_loaded = False
        latest_run = None
        ckpt = None
        run_config = None
        model = None
        
        for run in runs:
            c = get_best_checkpoint(run['path'])
            if not c: continue
                
            # Try to load the config
            run_config_path = get_config_path(run['path'])
            temp_config = OmegaConf.load(run_config_path) if run_config_path else base_config
            
            print(f"  Attempting to load checkpoint from run: {run['folder']}")
            try:
                # Need to clear CUDA memory before loading new model just in case
                torch.cuda.empty_cache()
                model_candidate = load_model_from_config(temp_config, c)
                
                # If we get here without an exception, it's NOT corrupted!
                model = model_candidate
                valid_model_loaded = True
                latest_run = run
                ckpt = c
                run_config = temp_config
                break 
                
            except Exception as e:
                print(f"  [Warning] Run {run['folder']} checkpoint is corrupted, skipping to older run... ({e})")
                continue
        
        if not valid_model_loaded:
            print(f"  [Skipping] No valid uncorrupted checkpoints found in any runs for {exp_name}")
            continue
            
        try:
            # Dynamically instantiate the exact dataset required by this model's config
            # (e.g. 15-channel DID vs 8-channel original) to prevent shape mismatches
            dataset_conf = run_config.data.params.validation
            if args.data_path:
                dataset_conf.params.data_path = args.data_path
            else:
                dataset_conf.params.data_path = fix_data_path(dataset_conf.params.data_path)
            dataset = instantiate_from_config(dataset_conf)
            
            # Force validation set augmentation to OFF to ensure GT and Predictions align
            if hasattr(dataset, 'augment'):
                dataset.augment = False
            
            sampler = DDIMSampler(model)
            
            # Run Inference Loop
            outdir = os.path.join(latest_run['path'], args.outdir_suffix)
            os.makedirs(outdir, exist_ok=True)
            
            model_metrics = []

            for i, idx in enumerate(tqdm(indices, desc=exp_name)):
                item = dataset[idx]
                x_raw = item['image']
                c_raw = item['cond']

                # Load raw un-normalized X for the domain panel visualization
                raw_x_np = None
                try:
                    chunk_idx, local_idx = dataset.indices[idx]
                    if getattr(dataset, 'is_hdf5', False):
                        # HDF5 backend (DID datasets)
                        import h5py
                        with h5py.File(dataset.files[0], 'r') as h5f:
                            raw_x_np = np.array(h5f['X'][local_idx], dtype=np.float32)
                    else:
                        # npz mmap backend (standard datasets)
                        raw_x_np = np.array(dataset.data_chunks_x[chunk_idx][local_idx], dtype=np.float32)
                except Exception:
                    pass  # domain panel will fall back gracefully

                # --- Exact old commit logic: put everything on CUDA, permute if channel-last ---
                if isinstance(x_raw, torch.Tensor):
                    x_gt = x_raw.unsqueeze(0).cuda()
                    cond = c_raw.unsqueeze(0).cuda()
                else:
                    x_gt = torch.from_numpy(x_raw).unsqueeze(0).cuda()
                    cond = torch.from_numpy(c_raw).unsqueeze(0).cuda()

                if x_gt.ndim == 4 and x_gt.shape[-1] < x_gt.shape[1]:
                    x_gt = x_gt.permute(0, 3, 1, 2)
                if cond.ndim == 4 and cond.shape[-1] < cond.shape[1]:
                    cond = cond.permute(0, 3, 1, 2)

                shape = (model.channels, model.image_size, model.image_size)

                t0 = time.time()
                with torch.no_grad():
                    samples_ddim, _ = sampler.sample(S=args.steps, conditioning=cond, batch_size=1, shape=shape, verbose=False)
                    x_samples = model.decode_first_stage(samples_ddim)
                t1 = time.time()
                inference_time = t1 - t0
                
                # --- Multi-channel handling (UK dataset support) ---
                pred_np_all = x_samples.cpu().numpy()[0] # (C, H, W)
                gt_np_all = x_gt.cpu().numpy()[0]
                
                # Un-normalize to [0, 2] per WindTransformer reference
                pred_np_all = pred_np_all + 1.0
                gt_np_all = gt_np_all + 1.0
                
                channel_names = ["Floor_Speed", "Floor_Turb", "Roof_Speed", "Roof_Turb"]
                
                for c in range(pred_np_all.shape[0]):
                    pred_np = pred_np_all[c]
                    gt_np = gt_np_all[c]
                    
                    c_name = channel_names[c] if c < len(channel_names) else f"Ch_{c}"
                    save_path = os.path.join(outdir, f"sample_{idx}_{c_name}.png")
                    
                    # Now we completely defer the plot to the function which also computes and returns metrics
                    plot_metrics = save_standardized_plot(gt_np, pred_np, save_path, x_raw_cond=raw_x_np, c_name=c_name)
                    
                    # Collect metrics for aggregation
                    metrics = {
                        "mae": plot_metrics['mae'],
                        "rmse": plot_metrics['rmse'],
                        "r2": plot_metrics['r2'],
                        "mape": plot_metrics['mape'],
                        "ssim": plot_metrics['ssim'],
                        "grad_corr": plot_metrics['grad_corr'],
                        "inference_time": inference_time,
                        "channel": c_name,
                        "sample_idx": idx
                    }
                    model_metrics.append(metrics)

            # Save Summary
            df = pd.DataFrame(model_metrics)
            if "channel" in df.columns:
                summary = df.groupby("channel").agg(['mean', 'std'])
            else:
                summary = df.agg(['mean', 'std'])
            summary.to_csv(os.path.join(outdir, "summary_metrics.csv"))
            
            # Flatten multi-index columns if they exist (for the master CSV)
            if isinstance(summary.columns, pd.MultiIndex):
                summary_flat = summary.xs('mean', axis=1, level=1)
                for channel_val, row_data in summary_flat.iterrows():
                    mean_row = row_data.to_dict()
                    mean_row['Experiment'] = f"{exp_name}_{channel_val}"
                    mean_row['Timestamp'] = latest_run['timestamp']
                    all_summaries.append(mean_row)
            else:
                mean_row = df.mean().to_dict()
                mean_row['Experiment'] = exp_name
                mean_row['Timestamp'] = latest_run['timestamp']
                all_summaries.append(mean_row)
            
            # Clean up Memory
            del model
            del sampler
            torch.cuda.empty_cache()
            
        except Exception as e:
            import traceback
            print(f"  [Error] Failed processing {exp_name}: {e}")
            traceback.print_exc()

    # Final Master CSV – append to existing file if present
    if all_summaries:
        master_df = pd.DataFrame(all_summaries)
        cols = ['Experiment', 'Timestamp'] + [c for c in master_df.columns if c not in ['Experiment', 'Timestamp']]
        master_df = master_df[cols]
        csv_path = "all_experiments_fast_summary.csv"
        if os.path.exists(csv_path):
            # Append without rewriting header
            master_df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            master_df.to_csv(csv_path, index=False)
        print("\nAppended to all_experiments_fast_summary.csv")

if __name__ == "__main__":
    main()
