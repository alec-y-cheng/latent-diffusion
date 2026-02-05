
import argparse
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
import torchmetrics
from tqdm import tqdm

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
    # Instantiate validation dataset from config
    print("Loading Validation Data...")
    dataset_conf = config.data.params.validation
    dataset = instantiate_from_config(dataset_conf)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)

    # Metrics
    # SSIM expects [0, 1] range usually? Or normalized.
    # Torchmetrics SSIM: expects tensors.
    # Torchmetrics 0.6.0 uses SSIM class. Newer uses StructSimIndexMeasure
    try:
        metric_ssim = torchmetrics.StructuralSimilarityIndexMeasure(data_range=2.0).cuda()
    except AttributeError:
        metric_ssim = torchmetrics.SSIM(data_range=2.0).cuda()
    metric_mse = torchmetrics.MeanSquaredError().cuda()

    all_rmse = []
    all_ssim = []
    
    count = 0
    
    print(f"Starting Inference (DDIM {args.steps} steps)...")
    
    # Visualization Helper
    def apply_cmap(t, cmap_name='viridis', vmin=None, vmax=None):
        import matplotlib.cm as cm
        x = t.detach().cpu().numpy().squeeze()
        # Handle 3-channel (RGB) if present? Here tensors are (H, W) or (1, H, W)
        # Check if 3 channels?
        if x.ndim == 3 and x.shape[0] == 3:
             x = x[0]
        
        if vmin is None: vmin = x.min()
        if vmax is None: vmax = x.max()
        
        norm_x = (x - vmin) / (vmax - vmin + 1e-6)
        cmapped = cm.get_cmap(cmap_name)(norm_x) # (H, W, 4)
        cmapped = cmapped[..., :3] 
        return (cmapped * 255).astype(np.uint8)

    save_count = 0
    max_save = 10
    
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
            
            x_samples = torch.clamp(x_samples, -1.0, 1.0)
            
            # Save Images
            if save_count < max_save:
                # Iterate over batch
                from PIL import Image
                for i in range(x_samples.shape[0]):
                    if save_count >= max_save: break
                    
                    img_sample = x_samples[i] # (1, H, W)
                    img_gt = x_gt[i]         # (1, H, W)
                    img_diff = img_sample - img_gt
                    
                    # Apply colormaps
                    vis_gt = apply_cmap(img_gt, 'viridis', -1, 1)
                    vis_pred = apply_cmap(img_sample, 'viridis', -1, 1)
                    vis_diff = apply_cmap(img_diff, 'RdBu_r', -1, 1)
                    
                    combined = np.concatenate([vis_gt, vis_pred, vis_diff], axis=1) # Side by Side
                    
                    save_path = os.path.join(args.outdir, f"sample_{save_count:03d}.png")
                    Image.fromarray(combined).save(save_path)
                    save_count += 1
            
            # Calculate Metrics
            # MSE
            mse = metric_mse(x_samples, x_gt)
            rmse = torch.sqrt(mse)
            all_rmse.append(rmse.item())
            
            # SSIM
            ssim_val = metric_ssim(x_samples, x_gt)
            all_ssim.append(ssim_val.item())
            
            count += z.shape[0]

    # Report
    mean_rmse = np.mean(all_rmse)
    mean_ssim = np.mean(all_ssim)
    
    print("="*30)
    print(f"Evaluation Results (N={count})")
    print(f"RMSE: {mean_rmse:.6f}")
    print(f"SSIM: {mean_ssim:.6f}")
    print("="*30)
    
    with open(os.path.join(args.outdir, "metrics.txt"), "w") as f:
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Samples: {count}\n")
        f.write(f"RMSE: {mean_rmse:.6f}\n")
        f.write(f"SSIM: {mean_ssim:.6f}\n")

if __name__ == "__main__":
    main()
