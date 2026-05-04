#!/usr/bin/env python3
"""
Full-val-set evaluation for pix2pix_hd_uk_roof checkpoints.

For each output channel (U, k, U_roof, k_roof) computes:
  - MAE, RMSE, MAPE
  - SSIM       (mean over val samples, restricted to visible domain)
  - GradCorr   (mean over val samples)
  - R² global  (aggregated across all pixels & samples — recommended for reporting)
  - R² per-sample (mean of per-sample R² — for comparison)

Writes results to <out_dir>/eval_metrics.json and prints a table.

Usage:
  python evaluate_uk_roof.py --ckpt <path_to_ckpt> [--data_path <npz>] [--batch 4]
                             [--out_dir <results_dir>] [--n_vis 5]
"""

import os, sys, json, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Reuse everything from the training module
from pix2pix_hd_uk_roof import (
    Pix2PixHDGenerator, EMA, XYMemmapDataset, build_data,
    save_pred_vs_true,
    X_CH, Y_CH, IMG_H, IMG_W,
    NGF, N_RESBLOCKS_G1, N_DOWNSAMPLE_G1, N_RESBLOCKS_G2,
    DATA_PATH_CLUSTER, DATA_PATH_AUG,
    CHANNEL_NAMES, CHANNEL_VMAXES,
    NUM_WORKERS, PIN_MEMORY, PERSISTENT_WORKERS, PREFETCH_FACTOR,
    seed_all, SEED,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_args():
    p = argparse.ArgumentParser(description="Full-val evaluation for pix2pix HD UK roof.")
    p.add_argument("--ckpt",      required=True, type=str,
                   help="Path to ckpt_best.pt (or any checkpoint with EMA state)")
    p.add_argument("--data_path", default=DATA_PATH_CLUSTER, type=str)
    p.add_argument("--augment",   action="store_true",
                   help="Use augmented dataset (matches training-time split)")
    p.add_argument("--batch",     default=4, type=int)
    p.add_argument("--out_dir",   default=None, type=str,
                   help="Output dir (default = ckpt parent dir / eval)")
    p.add_argument("--n_vis",     default=5, type=int,
                   help="Number of validation visualizations to save")
    p.add_argument("--no_local_enhancer", action="store_true")
    return p.parse_args()


@torch.no_grad()
def evaluate(args):
    seed_all(SEED)

    ckpt_path = args.ckpt
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    ckpt_dir = os.path.dirname(os.path.abspath(ckpt_path))
    run_dir  = os.path.dirname(ckpt_dir)
    if args.out_dir is None:
        args.out_dir = os.path.join(run_dir, "eval")
    os.makedirs(args.out_dir, exist_ok=True)
    vis_dir = os.path.join(args.out_dir, "images")
    os.makedirs(vis_dir, exist_ok=True)

    # --- Stats (saved alongside checkpoints) ---
    stats_path = os.path.join(ckpt_dir, "stats.pt")
    if not os.path.isfile(stats_path):
        raise FileNotFoundError(f"stats.pt not found in {ckpt_dir}")
    stats = torch.load(stats_path, map_location="cpu")
    x_mean = stats["x_mean"].to(device)
    x_std  = stats["x_std"].to(device)
    y_mean = stats["y_mean"].to(device)
    y_std  = stats["y_std"].to(device)
    print(f"Loaded stats from {stats_path}")

    # --- Data: same split as training ---
    data_path = args.data_path
    if not os.path.exists(data_path) and os.path.exists(DATA_PATH_AUG):
        data_path = DATA_PATH_AUG if args.augment else DATA_PATH_CLUSTER
    print(f"Data path: {data_path}")

    X_mm, Y_mm, train_idx, val_idx, _, _ = build_data(data_path, args.augment, args.out_dir)
    val_ds = XYMemmapDataset(X_mm, Y_mm, val_idx, augment=False)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                        persistent_workers=PERSISTENT_WORKERS,
                        prefetch_factor=PREFETCH_FACTOR)
    print(f"Val samples: {len(val_ds)}")

    # --- Model + EMA weights ---
    use_local_enhancer = not args.no_local_enhancer
    generator = Pix2PixHDGenerator(
        in_ch=X_CH, out_ch=Y_CH, ngf=NGF,
        use_local_enhancer=use_local_enhancer,
        n_resblocks_g1=N_RESBLOCKS_G1, n_downsample_g1=N_DOWNSAMPLE_G1,
        n_resblocks_g2=N_RESBLOCKS_G2,
    ).to(device)

    ck = torch.load(ckpt_path, map_location=device)
    if "ema_generator" in ck:
        generator.load_state_dict(ck["ema_generator"])
        src = "EMA"
    elif "generator" in ck:
        generator.load_state_dict(ck["generator"])
        src = "raw"
    else:
        raise KeyError("Checkpoint has neither 'ema_generator' nor 'generator' state")
    generator.eval()

    print(f"Loaded {src} weights from {ckpt_path} (epoch {ck.get('epoch', '?')}, "
          f"best_val={ck.get('best_val_loss', float('nan')):.6f})")

    # --- Circular domain mask (constant across samples) ---
    cy_m, cx_m = IMG_H // 2, IMG_W // 2
    rad_m      = min(IMG_H, IMG_W) // 2 - 5
    Yc, Xc     = np.ogrid[:IMG_H, :IMG_W]
    outside    = np.sqrt((Xc - cx_m) ** 2 + (Yc - cy_m) ** 2) >= rad_m  # (H,W) bool

    # --- Per-channel masking strategy (matches training viz) ---
    # ch 0,1 (U, k):       mask = outside | building_interior
    # ch 2,3 (U_roof, k_r): mask = outside | open_ground

    # --- Accumulators ---
    n_ch = Y_CH
    sum_abs   = np.zeros(n_ch, dtype=np.float64)
    sum_sq    = np.zeros(n_ch, dtype=np.float64)
    sum_y     = np.zeros(n_ch, dtype=np.float64)
    sum_y2    = np.zeros(n_ch, dtype=np.float64)
    n_pix     = np.zeros(n_ch, dtype=np.int64)

    abs_pct_sum   = np.zeros(n_ch, dtype=np.float64)  # for MAPE: sum of |err|/|y| where |y|>0.1
    abs_pct_count = np.zeros(n_ch, dtype=np.int64)

    # SSIM and GradCorr — compute per-image, average
    ssim_sum = np.zeros(n_ch, dtype=np.float64)
    grad_sum = np.zeros(n_ch, dtype=np.float64)
    n_imgs   = np.zeros(n_ch, dtype=np.int64)

    # Per-sample R² (mean of per-sample) — for comparison
    r2_per_sample_sum = np.zeros(n_ch, dtype=np.float64)
    r2_per_sample_n   = np.zeros(n_ch, dtype=np.int64)

    from skimage.metrics import structural_similarity as ssim_func

    n_vis_left = args.n_vis

    for xb, yb in tqdm(val_dl, desc="Eval"):
        xb_dev = xb.to(device, non_blocking=True)
        yb_dev = yb.to(device, non_blocking=True)

        x_n = (xb_dev - x_mean) / x_std
        pred_n = generator(x_n)
        pred = (pred_n * y_std + y_mean).cpu().numpy()  # (B, C, H, W) — denormalized
        true = yb.cpu().numpy()                          # (B, C, H, W)

        # clip predictions to be non-negative (matches viz behavior)
        pred = np.maximum(pred, 0.0)

        x_np = xb.cpu().numpy()  # (B, 8, H, W)

        B = pred.shape[0]
        for b in range(B):
            bldg_mask     = x_np[b, 4] > 0   # (H, W) — buildings
            open_gnd_mask = ~bldg_mask
            ch_extra = [bldg_mask, bldg_mask, open_gnd_mask, open_gnd_mask]

            for c in range(n_ch):
                hide        = outside | ch_extra[c]
                domain      = ~hide
                if not domain.any():
                    continue
                p_in = pred[b, c][domain]
                t_in = true[b, c][domain]
                err  = p_in - t_in
                abs_e = np.abs(err)

                sum_abs[c] += abs_e.sum()
                sum_sq[c]  += (err ** 2).sum()
                sum_y[c]   += t_in.sum()
                sum_y2[c]  += (t_in ** 2).sum()
                n_pix[c]   += t_in.size

                valid_mape = np.abs(t_in) > 0.1
                if valid_mape.any():
                    abs_pct_sum[c]   += (abs_e[valid_mape] / np.abs(t_in[valid_mape])).sum()
                    abs_pct_count[c] += valid_mape.sum()

                # Per-image SSIM (restricted to visible domain via SSIM map)
                gt_in_full = true[b, c]
                pr_in_full = pred[b, c]
                dr = max(t_in.max(), p_in.max()) - min(t_in.min(), p_in.min())
                if dr < 1e-8:
                    ssim_val = 1.0
                else:
                    _, ssim_map = ssim_func(gt_in_full, pr_in_full, data_range=dr, full=True)
                    ssim_val = float(ssim_map[domain].mean())
                ssim_sum[c] += ssim_val

                # GradCorr
                pdx = np.diff(pr_in_full, axis=1, prepend=pr_in_full[:, :1])
                pdy = np.diff(pr_in_full, axis=0, prepend=pr_in_full[:1, :])
                tdx = np.diff(gt_in_full, axis=1, prepend=gt_in_full[:, :1])
                tdy = np.diff(gt_in_full, axis=0, prepend=gt_in_full[:1, :])
                pg = np.concatenate([pdx.flatten(), pdy.flatten()])
                tg = np.concatenate([tdx.flatten(), tdy.flatten()])
                mm = np.concatenate([domain.flatten(), domain.flatten()])
                pg, tg = pg[mm], tg[mm]
                if np.std(pg) > 1e-8 and np.std(tg) > 1e-8:
                    grad_sum[c] += float(np.corrcoef(pg, tg)[0, 1])
                n_imgs[c] += 1

                # Per-sample R²
                ss_res_s = float((err ** 2).sum())
                ss_tot_s = float(((t_in - t_in.mean()) ** 2).sum())
                if ss_tot_s > 1e-12:
                    r2_per_sample_sum[c] += 1.0 - ss_res_s / ss_tot_s
                    r2_per_sample_n[c]   += 1

            # Save up to args.n_vis visualizations from the first batches
            if n_vis_left > 0:
                for c, (cn, cv) in enumerate(zip(CHANNEL_NAMES, CHANNEL_VMAXES)):
                    em = ch_extra[c]
                    save_pred_vs_true(
                        pred[b, c], true[b, c],
                        out_path=os.path.join(vis_dir, f"sample_{args.n_vis - n_vis_left:02d}_{cn}.png"),
                        cmap="viridis", interpolation="bilinear",
                        x=torch.from_numpy(x_np[b]), vmax=cv, channel_label=cn,
                        extra_mask=em,
                    )
                n_vis_left -= 1

    # --- Aggregate ---
    results = {}
    print()
    print(f"{'channel':<10} {'MAE':>10} {'RMSE':>10} {'MAPE%':>8} {'SSIM':>8} "
          f"{'GradCorr':>10} {'R2_global':>11} {'R2_persample':>13}")
    print("-" * 95)

    for c, cn in enumerate(CHANNEL_NAMES):
        n = max(n_pix[c], 1)
        mae   = sum_abs[c] / n
        rmse  = np.sqrt(sum_sq[c] / n)
        mape  = (abs_pct_sum[c] / max(abs_pct_count[c], 1)) * 100.0
        ssim  = ssim_sum[c] / max(n_imgs[c], 1)
        gradc = grad_sum[c] / max(n_imgs[c], 1)

        # Global R² (across all pixels & samples)
        mean_y = sum_y[c] / n
        ss_tot = sum_y2[c] - n * (mean_y ** 2)
        ss_res = sum_sq[c]
        r2_global = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

        r2_ps  = r2_per_sample_sum[c] / max(r2_per_sample_n[c], 1)

        print(f"{cn:<10} {mae:>10.4f} {rmse:>10.4f} {mape:>8.2f} {ssim:>8.4f} "
              f"{gradc:>10.4f} {r2_global:>11.4f} {r2_ps:>13.4f}")

        results[cn] = {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape_pct": float(mape),
            "ssim": float(ssim),
            "grad_corr": float(gradc),
            "r2_global": float(r2_global),
            "r2_per_sample_mean": float(r2_ps),
            "n_pixels": int(n_pix[c]),
            "n_images": int(n_imgs[c]),
        }

    out_json = os.path.join(args.out_dir, "eval_metrics.json")
    with open(out_json, "w") as f:
        json.dump({
            "checkpoint":  os.path.abspath(ckpt_path),
            "epoch":       int(ck.get("epoch", -1)),
            "val_samples": len(val_ds),
            "metrics":     results,
        }, f, indent=2)
    print(f"\nSaved: {out_json}")
    print(f"Sample images: {vis_dir}")


if __name__ == "__main__":
    evaluate(parse_args())
