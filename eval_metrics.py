import argparse
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim_func
except ImportError:
    print("Warning: skimage not found. SSIM will be disabled.")
    ssim_func = None

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if verbose:
        print("Missing keys:", m)
        print("Unexpected keys:", u)
    model.cuda()
    model.eval()
    return model

# --- Metric Helper Functions ---
def compute_gradient_correlation(pred, true, dmask=None):
    pred_dx = np.diff(pred, axis=1, prepend=pred[:, :1])
    pred_dy = np.diff(pred, axis=0, prepend=pred[:1, :])
    true_dx = np.diff(true, axis=1, prepend=true[:, :1])
    true_dy = np.diff(true, axis=0, prepend=true[:1, :])
    
    pred_grad = np.concatenate([pred_dx.flatten(), pred_dy.flatten()])
    true_grad = np.concatenate([true_dx.flatten(), true_dy.flatten()])
    
    if dmask is not None:
        mask_flat = np.concatenate([dmask.flatten(), dmask.flatten()])
        pred_grad = pred_grad[mask_flat]
        true_grad = true_grad[mask_flat]
    
    # Handle constant case to avoid NaN
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
    H, W = y_true.shape
    diff = y_pred - y_true
    
    # --- Domain Masking (Circular) ---
    # This assumes the domain is roughly circular centered in the image
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
        ax0.imshow(input_vis, cmap='viridis', origin='lower')
        ax0.set_title("Conditioning / Input")
    else:
        # Placeholder Wind Field similar to reference if no input provided
        step = 25
        Y_grid, X_grid = np.mgrid[0:H:step, 0:W:step]
        U = np.ones_like(X_grid, dtype=float) * 0.7
        V = np.ones_like(X_grid, dtype=float) * 0.7
        # Mask outside circle
        dist_grid = np.sqrt((X_grid - center_x)**2 + (Y_grid - center_y)**2)
        valid = dist_grid < radius
        U[~valid] = np.nan
        V[~valid] = np.nan
        
        ax0.imshow(np.ones((H, W)), cmap='Greys_r', vmin=0, vmax=1, origin='lower')
        ax0.quiver(X_grid, Y_grid, U, V, color='red', scale=30, width=0.004)
        ax0.set_title("Wind Input (Placeholder)")
    
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
    max_diff = max(abs(diff_masked.min()), abs(diff_masked.max()))
    im3 = ax3.imshow(diff, cmap='RdBu', vmin=-max(2.0, max_diff), vmax=max(2.0, max_diff), origin='lower')
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint ckpt")
    parser.add_argument("--outdir", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--steps", type=int, default=50, help="DDIM sampling steps")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of samples to evaluate")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load Config
    config = OmegaConf.load(args.config)
    
    # Load Model
    model = load_model_from_config(config, args.ckpt)
    sampler = DDIMSampler(model)

    # Load Validation Data
    print("Loading Validation Data...")
    try:
        dataset_conf = config.data.params.validation
        dataset = instantiate_from_config(dataset_conf)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    except Exception as e:
        print(f"Error loading dataset from config: {e}")
        return

    # Metrics Accumulators
    metrics_summary = {
        "mae": [], "rmse": [], "mape": [], "ssim": [], "grad_corr": []
    }
    
    count = 0
    save_count = 0
    max_save = 20
    
    print(f"Starting Inference (DDIM {args.steps} steps)...")
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if count >= args.num_samples:
                break
                
            # Prepare Input
            for k in batch:
                batch[k] = batch[k].cuda()
            
            z, c = model.get_input(batch, model.first_stage_key)
            
            # Sampling
            shape = (model.channels, model.image_size, model.image_size)
            samples_ddim, _ = sampler.sample(S=args.steps,
                                             conditioning=c,
                                             batch_size=z.shape[0],
                                             shape=shape,
                                             verbose=False)
            
            x_samples = model.decode_first_stage(samples_ddim)
            x_gt = batch[model.first_stage_key] 
            
            # Ensure x_gt is (B, C, H, W)
            if x_gt.ndim == 4 and x_gt.shape[-1] == 1:
                x_gt = x_gt.permute(0, 3, 1, 2)
            elif x_gt.ndim == 4 and x_gt.shape[-1] == model.channels: # channels last
                 x_gt = x_gt.permute(0, 3, 1, 2)

            # --- Processing loop for batch ---
            for i in range(x_samples.shape[0]):
                if count >= args.num_samples: break
                
                # Convert to numpy (H, W) - assuming single channel for CFD
                # If multi-channel, we might need to select one or average.
                pred_np = x_samples[i].cpu().numpy().squeeze()
                gt_np = x_gt[i].cpu().numpy().squeeze()
                
                # Handle potential channel dimension
                if pred_np.ndim == 3: pred_np = pred_np[0]
                if gt_np.ndim == 3: gt_np = gt_np[0]
                
                # Save visualization
                if save_count < max_save:
                    save_path = os.path.join(args.outdir, f"sample_{save_count:03d}.png")
                    sample_metrics = save_standardized_plot(gt_np, pred_np, save_path)
                    save_count += 1
                else:
                    # Just compute metrics without plot
                    # (Re-using the logic from save_standardized_plot but without plotting would be more efficient,
                    # but for now we can just call it or separate the logic. 
                    # For simplicity, let's call the helper but suppress plot if we separate it later.
                    # Actually, let's just inline the metric calc for speed if strictly needed, 
                    # or just accept the overhead for <100 samples.)
                    # Let's extract metric calc to be safe:
                    sample_metrics = save_standardized_plot(gt_np, pred_np, "temp.png")
                    if os.path.exists("temp.png"): os.remove("temp.png")

                # Accumulate
                for k in metrics_summary:
                    if k in sample_metrics:
                        metrics_summary[k].append(sample_metrics[k])
                
                count += 1

    # Report
    print("="*30)
    print(f"Evaluation Results (N={count})")
    for k, v in metrics_summary.items():
        if len(v) > 0:
            print(f"{k.upper()}: {np.mean(v):.6f}")
    print("="*30)
    
    with open(os.path.join(args.outdir, "metrics.txt"), "w") as f:
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Samples: {count}\n")
        for k, v in metrics_summary.items():
            if len(v) > 0:
                f.write(f"{k.upper()}: {np.mean(v):.6f}\n")

if __name__ == "__main__":
    main()
