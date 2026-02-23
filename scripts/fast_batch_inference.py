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

def get_experiment_groups(logs_dir, filter_str=None):
    groups = {}
    if not os.path.exists(logs_dir):
        print(f"Error: Logs directory '{logs_dir}' not found.")
        return {}

    for folder in os.listdir(logs_dir):
        path = os.path.join(logs_dir, folder)
        if not os.path.isdir(path):
            continue
            
        # Filter Logic
        if "autoencoder" in folder.lower():
            continue
        if filter_str:
            filters = [f.strip() for f in filter_str.split(',')]
            if not any(f in folder for f in filters):
                continue
        try:
            parts = folder.split('_')
            if len(parts) < 2: continue
            timestamp_str = parts[0]
            exp_name = "_".join(parts[1:])
            dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S")
            
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

def save_standardized_plot(y_true, y_pred, save_path, input_vis=None):
    if y_true.ndim > 2: y_true = y_true.squeeze()
    if y_pred.ndim > 2: y_pred = y_pred.squeeze()
    
    H, W = y_true.shape
    diff = y_pred - y_true
    
    # --- Domain Masking (Circular) ---
    center_y, center_x = H // 2, W // 2
    # radius = min(H, W) // 2 - 5
    # Use simpler radius logic to avoid issues with non-square
    radius = min(H, W) // 2
    Y_coords, X_coords = np.ogrid[:H, :W]
    dist = np.sqrt((X_coords - center_x)**2 + (Y_coords - center_y)**2)
    domain_mask = dist < radius
    
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
        data_range = max(y_true.max(), y_pred.max()) - min(y_true.min(), y_pred.min())
        if data_range == 0: data_range = 1.0
        ssim_val = ssim_func(y_true, y_pred, data_range=data_range)
    else:
        ssim_val = -1.0
        
    grad_corr = compute_gradient_correlation(y_pred, y_true, domain_mask)
    
    # --- Plot ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Panel 1: Input (Conditioning)
    ax0 = axes[0]
    if input_vis is not None:
        if input_vis.ndim > 2: input_vis = input_vis.squeeze()
        # Resize input_vis to H, W if needed for display
        if input_vis.shape != (H, W):
            # Simple resize via zoom not ideal, assume it's roughly correct or interpolated before call
            pass 
        ax0.imshow(input_vis, cmap='viridis', origin='lower')
        ax0.set_title("Input Condition")
    else:
        ax0.text(0.5, 0.5, "No Vis", ha='center')
        ax0.set_title("Input")
    
    ax0.set_xlim(0, W); ax0.set_ylim(0, H); ax0.set_aspect('equal')
    add_border(ax0)
    
    # Panel 2: GT
    ax1 = axes[1]
    im1 = ax1.imshow(y_true, origin='lower', cmap='viridis')
    ax1.set_title("Ground Truth")
    add_border(ax1)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Panel 3: Pred
    ax2 = axes[2]
    im2 = ax2.imshow(y_pred, origin='lower', cmap='viridis')
    ax2.set_title("Prediction")
    add_border(ax2)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Panel 4: Diff + Metrics
    ax3 = axes[3]
    # Center diff map around 0
    max_val = max(abs(diff.min()), abs(diff.max())) if diff.size > 0 else 1.0
    im3 = ax3.imshow(diff, cmap='RdBu', vmin=-max_val, vmax=max_val, origin='lower')
    ax3.set_title("Diff (Pred - GT)")
    
    metrics_text = (f"MAE:{mae:.3f} | RMSE:{rmse:.3f} | MAPE:{mape:.1f}%\n"
                    f"SSIM:{ssim_val:.3f} | GradCorr:{grad_corr:.3f}")
    ax3.set_xlabel(metrics_text, fontsize=9, family='monospace',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))
    
    add_border(ax3)
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

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

    # 1. Load Dataset ONCE
    print("Initializing Dataset (This happens only ONCE)...")
    base_config = OmegaConf.load(args.default_config)
    dataset_conf = base_config.data.params.validation
    
    if args.data_path:
        print(f"Overriding dataset path with: {args.data_path}")
        dataset_conf.params.data_path = args.data_path
    
    dataset = instantiate_from_config(dataset_conf)
    print(f"Dataset loaded. Length: {len(dataset)}")
    
    # Select indices shared across all models for fair comparison
    total_len = len(dataset)
    indices = np.random.choice(total_len, args.num_samples, replace=False)
    print(f"Selected indices for all models: {indices}")

    # Pre-load data items into CPU memory (if feasible) or just indices
    # To be safe and fast, let's keep it as indices access.

    # 2. Iterate Models
    groups = get_experiment_groups(args.logs, args.filter)
    all_summaries = []

    for exp_name, runs in groups.items():
        runs.sort(key=lambda x: x['timestamp'], reverse=True)
        # Try to find a valid checkpoint in the latest run first
        # But if the latest run failed (no checkpoints), we should check the next latest one!
        valid_run = None
        ckpt = None
        
        for run in runs:
            c = get_best_checkpoint(run['path'])
            if c:
                valid_run = run
                ckpt = c
                break
        
        if not valid_run:
            print(f"  [Skipping] No checkpoints found in any runs for {exp_name}")
            continue
            
        latest_run = valid_run # Use the valid one
        print(f"  Using checkpoint from run: {latest_run['folder']}")
            
        # Load Model Config (Model architecture might vary, so we load config per model)
        run_config_path = get_config_path(latest_run['path'])
        if not run_config_path:
            print("  No config found. Using default architecture.")
            run_config = base_config
        else:
            run_config = OmegaConf.load(run_config_path)

        try:
            # Re-instantiate model to clear previous state/memory
            model = load_model_from_config(run_config, ckpt)
            sampler = DDIMSampler(model)
            
            # Run Inference Loop
            outdir = os.path.join(latest_run['path'], args.outdir_suffix)
            os.makedirs(outdir, exist_ok=True)
            
            model_metrics = []
            
            for i, idx in enumerate(tqdm(indices, desc=exp_name)):
                item = dataset[idx]
                x_raw = item['image']
                c_raw = item['cond']
                
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
                
                # Metrics
                pred_np = x_samples.cpu().numpy()[0, 0]
                gt_np = x_gt.cpu().numpy()[0, 0]
                
                diff = pred_np - gt_np
                
                # Domain Mask Logic (Simplified)
                H, W = gt_np.shape
                center_y, center_x = H // 2, W // 2
                radius = min(H, W) // 2
                Y_coords, X_coords = np.ogrid[:H, :W]
                dist = np.sqrt((X_coords - center_x)**2 + (Y_coords - center_y)**2)
                domain_mask = dist < radius
                
                diff_masked = diff[domain_mask]
                abs_diff_masked = np.abs(diff_masked)
                
                mae = np.mean(abs_diff_masked)
                rmse = np.sqrt(np.mean(diff_masked**2))
                
                if ssim_func:
                    data_range = max(gt_np.max(), pred_np.max()) - min(gt_np.min(), pred_np.min())
                    if data_range == 0: data_range = 1.0
                    ssim_val = ssim_func(gt_np, pred_np, data_range=data_range)
                else:
                    ssim_val = -1.0
                    
                grad_corr = compute_gradient_correlation(pred_np, gt_np, domain_mask)
                
                model_metrics.append({
                    "mae": mae,
                    "rmse": rmse,
                    "ssim": ssim_val,
                    "grad_corr": grad_corr,
                    "inference_time": inference_time
                })
                
                # Optional: Save Image (First sample only)
                if i == 0:
                    # Prepare Visualization inputs
                    # Condition: If we can assume Channel 0 is mask, or visualize something meaningful
                    if cond.shape[1] > 0:
                         bg_low = cond.cpu().numpy()[0, 0]
                         import cv2
                         H_disp, W_disp = pred_np.shape[0], pred_np.shape[1]
                         bg_vis = cv2.resize(bg_low, (W_disp, H_disp), interpolation=cv2.INTER_NEAREST)
                    else:
                         bg_vis = None
                         
                    save_path = os.path.join(outdir, f"sample_{idx}.png")
                    save_standardized_plot(gt_np, pred_np, save_path, input_vis=bg_vis)

            # Save Summary
            df = pd.DataFrame(model_metrics)
            summary = df.agg(['mean', 'std'])
            summary.to_csv(os.path.join(outdir, "summary_metrics.csv"))
            
            mean_row = df.mean().to_dict()
            mean_row['Experiment'] = exp_name
            mean_row['Timestamp'] = latest_run['timestamp']
            all_summaries.append(mean_row)
            
            # Clean up Memory
            del model
            del sampler
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"  [Error] Failed processing {exp_name}: {e}")

    # Final Master CSV
    if all_summaries:
        master_df = pd.DataFrame(all_summaries)
        cols = ['Experiment', 'Timestamp'] + [c for c in master_df.columns if c not in ['Experiment', 'Timestamp']]
        master_df = master_df[cols]
        master_df.to_csv("all_experiments_fast_summary.csv", index=False)
        print("\nSaved all_experiments_fast_summary.csv")

if __name__ == "__main__":
    main()
