import argparse
import csv
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.append(PROJECT_ROOT)
sys.path.append(SCRIPTS_DIR)

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from fast_batch_inference import (
    compute_gradient_correlation,
    fix_data_path,
    get_best_checkpoint,
    get_config_path,
    get_experiment_groups,
    load_model_from_config,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None


CHANNEL_NAMES = ["Floor_Speed", "Floor_Turb", "Roof_Speed", "Roof_Turb"]
CH_LOCAL_X = 4
CH_LOCAL_Y = 5
CH_WIND_X = 6
CH_WIND_Y = 7


# Condition tensors are C,H,W. Output channels below are scalar H,W fields.
def _flip_h(x):
    y = np.flip(x, axis=2).copy()
    if y.shape[0] > CH_LOCAL_X:
        y[CH_LOCAL_X] *= -1.0
    if y.shape[0] > CH_WIND_X:
        y[CH_WIND_X] *= -1.0
    return y


def _flip_v(x):
    y = np.flip(x, axis=1).copy()
    if y.shape[0] > CH_LOCAL_Y:
        y[CH_LOCAL_Y] *= -1.0
    if y.shape[0] > CH_WIND_Y:
        y[CH_WIND_Y] *= -1.0
    return y


def _transpose_xy(x):
    y = np.swapaxes(x, 1, 2).copy()
    if y.shape[0] > CH_LOCAL_Y:
        old = y.copy()
        y[CH_LOCAL_X], y[CH_LOCAL_Y] = old[CH_LOCAL_Y], old[CH_LOCAL_X]
    if y.shape[0] > CH_WIND_Y:
        old = y.copy()
        y[CH_WIND_X], y[CH_WIND_Y] = old[CH_WIND_Y], old[CH_WIND_X]
    return y


def _chw_spatial_h(x):
    return np.flip(x, axis=2).copy()


def _chw_spatial_v(x):
    return np.flip(x, axis=1).copy()


def _chw_spatial_t(x):
    return np.swapaxes(x, 1, 2).copy()


def _spatial_h(x):
    return np.flip(x, axis=1).copy()


def _spatial_v(x):
    return np.flip(x, axis=0).copy()


def _spatial_t(x):
    return np.swapaxes(x, 0, 1).copy()


TRANSFORMS = {
    "hflip": {
        "cond": lambda x: _flip_h(x),
        "latent": lambda z: _chw_spatial_h(z),
        "out": lambda y: _spatial_h(y),
        "inverse_out": lambda y: _spatial_h(y),
    },
    "vflip": {
        "cond": lambda x: _flip_v(x),
        "latent": lambda z: _chw_spatial_v(z),
        "out": lambda y: _spatial_v(y),
        "inverse_out": lambda y: _spatial_v(y),
    },
    "rot180": {
        "cond": lambda x: _flip_v(_flip_h(x)),
        "latent": lambda z: _chw_spatial_v(_chw_spatial_h(z)),
        "out": lambda y: _spatial_v(_spatial_h(y)),
        "inverse_out": lambda y: _spatial_h(_spatial_v(y)),
    },
    "transpose": {
        "cond": lambda x: _transpose_xy(x),
        "latent": lambda z: _chw_spatial_t(z),
        "out": lambda y: _spatial_t(y),
        "inverse_out": lambda y: _spatial_t(y),
    },
    "rot90": {
        "cond": lambda x: _flip_v(_transpose_xy(x)),
        "latent": lambda z: _chw_spatial_v(_chw_spatial_t(z)),
        "out": lambda y: _spatial_v(_spatial_t(y)),
        "inverse_out": lambda y: _spatial_t(_spatial_v(y)),
    },
    "rot270": {
        "cond": lambda x: _flip_h(_transpose_xy(x)),
        "latent": lambda z: _chw_spatial_h(_chw_spatial_t(z)),
        "out": lambda y: _spatial_h(_spatial_t(y)),
        "inverse_out": lambda y: _spatial_t(_spatial_h(y)),
    },
    "anti_transpose": {
        "cond": lambda x: _flip_v(_flip_h(_transpose_xy(x))),
        "latent": lambda z: _chw_spatial_v(_chw_spatial_h(_chw_spatial_t(z))),
        "out": lambda y: _spatial_v(_spatial_h(_spatial_t(y))),
        "inverse_out": lambda y: _spatial_t(_spatial_h(_spatial_v(y))),
    },
}


def tensor_to_chw_np(value):
    if isinstance(value, torch.Tensor):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)
    if arr.ndim == 3 and arr.shape[-1] < arr.shape[0]:
        arr = np.transpose(arr, (2, 0, 1))
    return arr.astype(np.float32, copy=False)


def chw_to_model_tensor(chw, device):
    return torch.from_numpy(chw).unsqueeze(0).to(device=device, dtype=torch.float32)


def predict(model, sampler, cond_chw, steps, x_t_chw=None):
    device = next(model.parameters()).device
    cond = chw_to_model_tensor(cond_chw, device)
    shape = (model.channels, model.image_size, model.image_size)
    x_t = None
    if x_t_chw is not None:
        x_t = chw_to_model_tensor(x_t_chw, device)
    with torch.no_grad():
        samples, _ = sampler.sample(S=steps, conditioning=cond, batch_size=1, shape=shape, verbose=False, x_T=x_t)
        decoded = model.decode_first_stage(samples)
    return decoded.detach().cpu().numpy()[0].astype(np.float32)


def make_initial_noise(model, seed):
    device = next(model.parameters()).device
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    shape = (model.channels, model.image_size, model.image_size)
    noise = torch.randn((1, *shape), generator=generator, device=device)
    return noise.detach().cpu().numpy()[0].astype(np.float32)


def metric_row(base, other):
    diff = other - base
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_abs = float(np.max(np.abs(diff)))
    grad_corr = float(compute_gradient_correlation(other, base))
    denom = float(np.sqrt(np.mean(base ** 2)) + 1e-8)
    rel_rmse = rmse / denom
    return {
        "mae": mae,
        "rmse": rmse,
        "relative_rmse": rel_rmse,
        "max_abs": max_abs,
        "grad_corr": grad_corr,
    }


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    grouped = {}
    for row in rows:
        key = (row["experiment"], row["transform"], row["channel"])
        grouped.setdefault(key, []).append(row)

    summary = []
    metric_names = ["mae", "rmse", "relative_rmse", "max_abs", "grad_corr", "inference_time"]
    for (experiment, transform, channel), items in sorted(grouped.items()):
        out = {
            "experiment": experiment,
            "transform": transform,
            "channel": channel,
            "n": len(items),
        }
        for metric in metric_names:
            vals = np.asarray([float(item[metric]) for item in items], dtype=np.float64)
            out[f"{metric}_mean"] = float(vals.mean())
            out[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary.append(out)

    overall = {}
    for row in rows:
        key = (row["experiment"], row["transform"], "ALL")
        overall.setdefault(key, []).append(row)
    for (experiment, transform, channel), items in sorted(overall.items()):
        out = {
            "experiment": experiment,
            "transform": transform,
            "channel": channel,
            "n": len(items),
        }
        for metric in metric_names:
            vals = np.asarray([float(item[metric]) for item in items], dtype=np.float64)
            out[f"{metric}_mean"] = float(vals.mean())
            out[f"{metric}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        summary.append(out)
    return summary


def save_example_plot(outdir, exp_name, sample_idx, base_pred, aug_back_by_transform, metrics_by_transform):
    if plt is None:
        return
    example_dir = os.path.join(outdir, "examples")
    os.makedirs(example_dir, exist_ok=True)

    display = base_pred + 1.0
    aug_display = {name: pred + 1.0 for name, pred in aug_back_by_transform.items()}
    columns = ["original"] + list(aug_display.keys())

    for ch in range(base_pred.shape[0]):
        channel = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"Ch_{ch}"
        fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 4), constrained_layout=True)
        if len(columns) == 1:
            axes = [axes]
        stack_vals = [display[ch]] + [aug_display[name][ch] for name in aug_display]
        vmin = min(float(v.min()) for v in stack_vals)
        vmax = max(float(v.max()) for v in stack_vals)
        for ax, col in zip(axes, columns):
            if col == "original":
                img = display[ch]
                title = "Original"
            else:
                img = aug_display[col][ch]
                m = metrics_by_transform[(col, channel)]
                title = f"{col}\nRMSE {m['rmse']:.4f} | GC {m['grad_corr']:.3f}"
            im = ax.imshow(img, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.colorbar(im, ax=axes, shrink=0.75)
        fig.suptitle(f"{exp_name} sample {sample_idx} {channel}")
        path = os.path.join(example_dir, f"{exp_name}_sample-{sample_idx}_{channel}.png")
        fig.savefig(path, dpi=140)
        plt.close(fig)


def load_dataset_from_config(config, split, data_path=None):
    if split == "train":
        dataset_conf = config.data.params.train
    elif split in ("val", "validation"):
        dataset_conf = config.data.params.validation
    else:
        raise ValueError("split must be train or validation")

    if data_path:
        dataset_conf.params.data_path = data_path
    else:
        dataset_conf.params.data_path = fix_data_path(dataset_conf.params.data_path)
    dataset = instantiate_from_config(dataset_conf)
    if hasattr(dataset, "augment"):
        dataset.augment = False
    return dataset


def choose_indices(dataset_len, max_samples, seed):
    if max_samples is None or max_samples <= 0 or max_samples >= dataset_len:
        return np.arange(dataset_len)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(dataset_len, size=max_samples, replace=False))


def choose_example_indices(indices, num_examples, seed):
    if num_examples <= 0 or len(indices) == 0:
        return set()
    n = min(num_examples, len(indices))
    rng = np.random.default_rng(seed + 1009)
    return set(rng.choice(indices, size=n, replace=False).astype(int).tolist())


def main():
    parser = argparse.ArgumentParser(description="Check LDM augmentation consistency.")
    parser.add_argument("--logs", default="logs")
    parser.add_argument("--filter", default="uk_grad_corr_low,uk_pinns_baseline")
    parser.add_argument("--data_path", default="data/uk_roof_dataset.h5")
    parser.add_argument("--split", default="train", choices=["train", "validation", "val"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--max_samples", type=int, default=0, help="0 means all samples in split")
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--transforms", default="hflip,vflip,rot180,transpose,rot90,rot270,anti_transpose")
    parser.add_argument("--outdir_suffix", default="augmentation_consistency")
    args = parser.parse_args()

    transform_names = [name.strip() for name in args.transforms.split(",") if name.strip()]
    unknown = [name for name in transform_names if name not in TRANSFORMS]
    if unknown:
        raise ValueError(f"Unknown transforms: {unknown}. Valid: {sorted(TRANSFORMS)}")

    groups = get_experiment_groups(args.logs, args.filter)
    if not groups:
        raise RuntimeError(f"No experiment groups found under {args.logs} with filter {args.filter}")

    for exp_name, runs in groups.items():
        runs.sort(key=lambda x: x["timestamp"], reverse=True)
        run = runs[0]
        ckpt = get_best_checkpoint(run["path"])
        config_path = get_config_path(run["path"])
        if not ckpt or not config_path:
            print(f"[Skipping] {exp_name}: missing checkpoint or config")
            continue

        print(f"\n=== {exp_name} ===")
        print(f"Run: {run['path']}")
        print(f"Checkpoint: {ckpt}")
        print(f"Config: {config_path}")

        config = OmegaConf.load(config_path)
        model = load_model_from_config(config, ckpt)
        sampler = DDIMSampler(model)
        dataset = load_dataset_from_config(config, args.split, args.data_path)
        indices = choose_indices(len(dataset), args.max_samples, args.seed)
        example_indices = choose_example_indices(indices, args.num_examples, args.seed)

        outdir = os.path.join(run["path"], args.outdir_suffix)
        os.makedirs(outdir, exist_ok=True)

        rows = []
        for sample_idx in tqdm(indices, desc=f"{exp_name} {args.split}"):
            item = dataset[int(sample_idx)]
            cond_chw = tensor_to_chw_np(item["cond"])
            base_noise = make_initial_noise(model, args.seed + int(sample_idx))
            base_pred = predict(model, sampler, cond_chw, args.steps, x_t_chw=base_noise)

            aug_back_by_transform = {}
            metrics_by_transform = {}
            for transform_name in transform_names:
                transform = TRANSFORMS[transform_name]
                cond_aug = transform["cond"](cond_chw)
                noise_aug = transform["latent"](base_noise)
                t0 = time.time()
                pred_aug = predict(model, sampler, cond_aug, args.steps, x_t_chw=noise_aug)
                inference_time = time.time() - t0
                pred_back = np.stack([transform["inverse_out"](pred_aug[ch]) for ch in range(pred_aug.shape[0])], axis=0)
                aug_back_by_transform[transform_name] = pred_back

                for ch in range(base_pred.shape[0]):
                    channel = CHANNEL_NAMES[ch] if ch < len(CHANNEL_NAMES) else f"Ch_{ch}"
                    metrics = metric_row(base_pred[ch], pred_back[ch])
                    metrics["experiment"] = exp_name
                    metrics["sample_idx"] = int(sample_idx)
                    metrics["transform"] = transform_name
                    metrics["channel"] = channel
                    metrics["inference_time"] = inference_time
                    rows.append(metrics)
                    metrics_by_transform[(transform_name, channel)] = metrics

            if int(sample_idx) in example_indices:
                save_example_plot(outdir, exp_name, int(sample_idx), base_pred, aug_back_by_transform, metrics_by_transform)

        per_sample_fields = [
            "experiment", "sample_idx", "transform", "channel",
            "mae", "rmse", "relative_rmse", "max_abs", "grad_corr", "inference_time",
        ]
        write_csv(os.path.join(outdir, "per_sample_consistency.csv"), rows, per_sample_fields)

        summary = summarize(rows)
        summary_fields = list(summary[0].keys()) if summary else []
        if summary:
            write_csv(os.path.join(outdir, "summary_consistency.csv"), summary, summary_fields)
        print(f"Saved results to {outdir}")

        del model
        del sampler
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
