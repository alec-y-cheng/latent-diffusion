
import argparse
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

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
        
        with torch.no_grad():
            samples_ddim, _ = sampler.sample(S=args.steps,
                                             conditioning=cond,
                                             batch_size=1,
                                             shape=shape,
                                             verbose=False)
            x_samples = model.decode_first_stage(samples_ddim)
            
        # Post-process (Single Sample)
        pred_np = x_samples.cpu().numpy()[0, 0] # (H, W) or (C, H, W)? Assuming single channel target
        gt_np = x_gt.cpu().numpy()[0, 0]
        
        # Visualization Input (Conditioning)
        # Attempt to visualize Vector Field (U, V) if present (Channels 4, 5)
        do_quiver = False
        if cond.shape[0] >= 6:
            # U = Channel 4, V = Channel 5 (from cfd_data.py flip logic)
            u_low = cond.cpu().numpy()[0, 4]
            v_low = cond.cpu().numpy()[0, 5]
            do_quiver = True
            
            # Mask (Channel 0) for background
            bg_low = cond.cpu().numpy()[0, 0]
        elif cond.shape[0] > 0:
            bg_low = cond.cpu().numpy()[0, 0]
        else:
            bg_low = np.zeros_like(pred_np)

        # Resize for visualization (High Res)
        import cv2
        H_disp, W_disp = pred_np.shape[0], pred_np.shape[1]
        bg_vis = cv2.resize(bg_low, (W_disp, H_disp), interpolation=cv2.INTER_NEAREST)
        
        save_path = os.path.join(args.outdir, f"test_sample_{i:03d}_idx_{idx}.png")
        
        # We need to pass data to plotting function or handle plot here
        # Let's update save_standardized_plot to accept vector field data
        
        # --- Metrics ---
        diff = pred_np - gt_np # Re-calc here for scope
        
        # --- Plot ---
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # Panel 1: Input (Vector Field)
        ax0 = axes[0]
        ax0.imshow(bg_vis, cmap='gray', origin='lower', alpha=0.5) # Background mask
        if do_quiver:
            # Downsample for cleaner quiver (every 16th pixel maybe?)
            step = 16
            Y, X = np.mgrid[0:H_disp:step, 0:W_disp:step]
            
            # Resize U, V to match display size first
            u_vis = cv2.resize(u_low, (W_disp, H_disp), interpolation=cv2.INTER_LINEAR)
            v_vis = cv2.resize(v_low, (W_disp, H_disp), interpolation=cv2.INTER_LINEAR)
            
            U = u_vis[::step, ::step]
            V = v_vis[::step, ::step]
            
            ax0.quiver(X, Y, U, V, color='red', scale=20, width=0.005)
            ax0.set_title("Input (Wind Vectors)")
        else:
            ax0.set_title("Input (Mask)")
            
        ax0.set_xlim(0, W_disp); ax0.set_ylim(0, H_disp); ax0.set_aspect('equal')
        add_border(ax0)
        
        # Panel 2: GT
        ax1 = axes[1]
        im1 = ax1.imshow(gt_np, origin='lower', cmap='viridis')
        ax1.set_title("Ground Truth")
        add_border(ax1)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Panel 3: Pred
        ax2 = axes[2]
        im2 = ax2.imshow(pred_np, origin='lower', cmap='viridis')
        ax2.set_title("Prediction")
        add_border(ax2)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # Panel 4: Diff + Metrics
        # Recalculate metrics locally as previous code did
        # (This block duplicates some metric logic but keeps flow simple)
        # We rely on metrics calculated before... wait, need to pass them or recalc
        # Re-using variables from scope: mae, rmse, etc. calculated above?
        # AHH, save_standardized_plot does metrics internaly.
        # I should modify save_standardized_plot instead... 
        
        # ... actually, refactoring save_standardized_plot is cleaner.
        # But for now, let's just use the existing function and pass the quiver data via a hack 
        # OR inline the plotting here.
        # Inline is safer to match user request quickly.
        
        # Re-calc metrics (copy-paste logic for safety)
        diff = pred_np - gt_np
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
        gt_masked = gt_np[domain_mask]
        gt_abs = np.abs(gt_masked)
        valid_for_mape = gt_abs > 0.1
        if np.any(valid_for_mape):
            mape = np.mean(abs_diff_masked[valid_for_mape] / gt_abs[valid_for_mape]) * 100.0
        else:
            mape = 0.0
        if ssim_func:
            data_range = max(gt_np.max(), pred_np.max()) - min(gt_np.min(), pred_np.min())
            if data_range == 0: data_range = 1.0
            ssim_val = ssim_func(gt_np, pred_np, data_range=data_range)
        else:
            ssim_val = -1.0
        grad_corr = compute_gradient_correlation(pred_np, gt_np, domain_mask)
        
        # Panel 4 again
        ax3 = axes[3]
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
        
        
        # Collect metrics for aggregation
        metrics = {
            "mae": mae,
            "rmse": rmse,
            "mape": mape,
            "ssim": ssim_val,
            "grad_corr": grad_corr
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
