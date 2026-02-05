
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
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if count >= args.num_samples:
                break
                
            # Prepare Input
            # move batch to device
            for k in batch:
                batch[k] = batch[k].cuda()
            
            # Get conditioning
            # LDM expects specific handling. model.get_input handles keys.
            # But get_input returns specific tensors.
            # We need to manually replicate get_input logic or call it
            # c, xc, ... = model.get_input(batch, model.first_stage_key)
            # But we want to SAMPLE.
            
            # c = conditioning (latent)
            # z = latent target (for autoencoder)
            z, c = model.get_input(batch, model.first_stage_key)
            
            # Sampling
            # shape of latent: (B, 4, 64, 64). (From config: 4 channels, 64x64)
            shape = (model.channels, model.image_size, model.image_size)
            samples_ddim, _ = sampler.sample(S=args.steps,
                                             conditioning=c,
                                             batch_size=z.shape[0],
                                             shape=shape,
                                             verbose=False)
            
            # Decode Latents to Image Space
            x_samples = model.decode_first_stage(samples_ddim)
            x_gt = model.decode_first_stage(z) # Reconstruct GT from latent (fair comparison) or use Raw batch?
            # Using decoded GT is better if we evaluate Generation independent of AE loss.
            # But user wants "Pred vs GT".
            # Usually we compare against RAW GT (batch['image']).
            # But 'batch['image']' is 512x512.
            # model.decode might be 512x512.
            
            # Let's compare against Raw GT to capture AE loss too?
            x_raw_gt = batch[model.first_stage_key] # Inputs [-1, 1]
            
            # Clamp to [-1, 1]
            x_samples = torch.clamp(x_samples, -1.0, 1.0)
            
            # Calculate Metrics
            # MSE
            mse = metric_mse(x_samples, x_raw_gt)
            rmse = torch.sqrt(mse)
            all_rmse.append(rmse.item())
            
            # SSIM
            # Torchmetrics expects (B, C, H, W)
            ssim_val = metric_ssim(x_samples, x_raw_gt)
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
    
    # Save to file
    with open(os.path.join(args.outdir, "metrics.txt"), "w") as f:
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Samples: {count}\n")
        f.write(f"RMSE: {mean_rmse:.6f}\n")
        f.write(f"SSIM: {mean_ssim:.6f}\n")

if __name__ == "__main__":
    main()
