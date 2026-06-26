#!/bin/bash
#SBATCH --job-name=inspect_uk_h5
#SBATCH --account=gts-pkastner3
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=logs/inspect_uk_h5_%j.out

set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-}"
if [[ -z "$PROJECT_ROOT" || ! -f "$PROJECT_ROOT/scripts/slurm_env.sh" ]]; then
  PROJECT_ROOT="/storage/scratch1/7/acheng324/latent-diffusion"
fi
cd "$PROJECT_ROOT"

mkdir -p logs

source "$PROJECT_ROOT/scripts/slurm_env.sh"
activate_ldm_env

# Override like:
#   DATA_FILE=/path/to/uk_roof_dataset.h5 sbatch scripts/inspect_uk_roof_h5.slurm
DATA_FILE="${DATA_FILE:-data/uk_roof_dataset.h5}"

if [[ ! -f "$DATA_FILE" ]]; then
  echo "ERROR: HDF5 file not found: $DATA_FILE" >&2
  exit 2
fi

python - "$DATA_FILE" <<'PY'
import sys

try:
    import h5py
except ImportError as exc:
    raise SystemExit(
        "ERROR: h5py is not installed in this Python environment. "
        "Activate your LDM/CFD environment or install h5py, then rerun."
    ) from exc

import numpy as np


INPUT_CHANNELS = [
    "SDF",
    "Building Height",
    "Relative Height Z",
    "Background Wind Ratio U/Uref",
    "Local X Coordinate",
    "Local Y Coordinate",
    "Wind Direction Sin",
    "Wind Direction Cos",
]

OUTPUT_CHANNELS = [
    "u_ground",
    "u_roof",
    "tke_ground",
    "tke_roof",
]


def print_tree(h5):
    print("\n=== HDF5 hierarchy ===")

    def visitor(name, obj):
        indent = "  " * name.count("/")
        short = name.rsplit("/", 1)[-1]
        if isinstance(obj, h5py.Group):
            print(f"{indent}{short}/")
        elif isinstance(obj, h5py.Dataset):
            print(f"{indent}{short}: shape={obj.shape}, dtype={obj.dtype}")

    if not h5.keys():
        print("(empty file)")
    else:
        for key, obj in h5.items():
            if isinstance(obj, h5py.Group):
                print(f"{key}/")
                obj.visititems(lambda name, child: visitor(f"{key}/{name}", child))
            else:
                print(f"{key}: shape={obj.shape}, dtype={obj.dtype}")


def collect_datasets(h5):
    datasets = {}

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets[name] = obj

    h5.visititems(visitor)
    return datasets


def guess_role(name, dataset):
    lower = name.lower()
    if any(token in lower for token in ("input", "cond", "x")):
        return "input"
    if any(token in lower for token in ("target", "output", "label", "y")):
        return "output"
    if dataset.ndim >= 3:
        channels = dataset.shape[-1] if dataset.shape[-1] <= 32 else dataset.shape[1]
        if channels == 8:
            return "input"
        if channels in (1, 4):
            return "output"
    return "unknown"


def channel_axis_for_shape(shape):
    if len(shape) < 3:
        return None
    if len(shape) >= 4 and shape[1] <= 32:
        return 1
    if shape[-1] <= 32:
        return -1
    if len(shape) == 3 and shape[0] <= 32:
        return 0
    return None


def sample_channel_axis(dataset_axis, dataset_ndim):
    if dataset_axis is None:
        return None
    if dataset_ndim >= 4 and dataset_axis > 0:
        return dataset_axis - 1
    return dataset_axis


def first_sample(dataset):
    if dataset.ndim == 0:
        return dataset[()]
    if dataset.ndim >= 4:
        return dataset[0]
    return dataset[...]


def summarize_dataset(name, dataset):
    role = guess_role(name, dataset)
    dataset_axis = channel_axis_for_shape(dataset.shape)
    axis = sample_channel_axis(dataset_axis, dataset.ndim)
    sample = first_sample(dataset)

    print(f"\n=== Dataset: {name} ===")
    print(f"role_guess: {role}")
    print(f"shape: {dataset.shape}")
    print(f"dtype: {dataset.dtype}")
    print(f"channel_axis_in_dataset: {dataset_axis}")
    print(f"attrs: {dict(dataset.attrs)}")

    if not np.issubdtype(dataset.dtype, np.number):
        preview = dataset[0] if dataset.ndim else dataset[()]
        print(f"non_numeric_preview: {preview!r}")
        return

    arr = np.asarray(sample)
    print(f"first_sample_shape: {arr.shape}")
    print(f"first_sample_min/max/mean: {arr.min():.6g} / {arr.max():.6g} / {arr.mean():.6g}")

    if axis is None:
        flat = arr.reshape(-1)
        print(f"first_values: {flat[:12]}")
        return

    names = INPUT_CHANNELS if role == "input" else OUTPUT_CHANNELS if role == "output" else []
    channels = arr.shape[axis]
    print(f"channel_axis_in_sample: {axis}")
    print(f"num_channels: {channels}")

    for ch in range(channels):
        label = names[ch] if ch < len(names) else f"channel_{ch}"
        channel = np.take(arr, ch, axis=axis)
        flat = channel.reshape(-1)
        center = tuple(s // 2 for s in channel.shape)
        print(
            f"  [{ch}] {label}: "
            f"shape={channel.shape}, "
            f"min={channel.min():.6g}, max={channel.max():.6g}, mean={channel.mean():.6g}, "
            f"center={channel[center]:.6g}, first_values={flat[:6]}"
        )


path = sys.argv[1]
print(f"Inspecting: {path}")

with h5py.File(path, "r") as h5:
    print_tree(h5)
    datasets = collect_datasets(h5)
    print(f"\nFound {len(datasets)} datasets.")
    for name, dataset in datasets.items():
        summarize_dataset(name, dataset)
PY
