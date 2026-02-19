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
        if filter_str and filter_str not in folder:
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
    ckpts.sort(key=os.path.getmtime, reverse=True)
    return ckpts[0]

def get_config_path(folder):
    cfg = os.path.join(folder, "configs", "project.yaml")
    if os.path.exists(cfg): return cfg
    cfg_dir = os.path.join(folder, "configs")
    if os.path.exists(cfg_dir):
        yamls = glob.glob(os.path.join(cfg_dir, "*.yaml"))
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

def load_model_from_config(config, ckpt):
    print(f"Loading model state from {ckpt}...")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
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
        latest_run = runs[0]
        
        print(f"\n--- Processing: {exp_name} ---")
        
        ckpt = get_best_checkpoint(latest_run['path'])
        if not ckpt:
            print("  No checkpoint found. Skipping.")
            continue
            
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
                
                # Optional: Save Image (Every 10th sample specific to model?)
                # To save time, maybe skip saving images in fast mode or only save first 1
                if i == 0:
                    # Save just one example to prove it worked
                    plt.figure()
                    plt.imshow(pred_np, cmap='viridis')
                    plt.title(f"Pred (SSIM: {ssim_val:.2f})")
                    plt.savefig(os.path.join(outdir, f"sample_{idx}.png"))
                    plt.close()

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
