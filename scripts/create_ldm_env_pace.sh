#!/bin/bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

ENV_NAME="${LDM_ENV_NAME:-ldm}"
ENV_PATH="${LDM_ENV_PATH:-/storage/scratch1/7/acheng324/conda/envs/$ENV_NAME}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/storage/scratch1/7/acheng324/conda/pkgs}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/storage/scratch1/7/acheng324/.cache}"
export CONDA_PKGS_DIRS
export XDG_CACHE_HOME
export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"

if module load mamba/1.4.9 2>/dev/null; then
  eval "$(conda shell.bash hook)"
  CREATE_CMD=(mamba env create -p "$ENV_PATH" -f environment.pace-h200.yaml)
else
  module load anaconda3/2023.03
  eval "$(conda shell.bash hook)"
  CREATE_CMD=(conda env create --solver classic -p "$ENV_PATH" -f environment.pace-h200.yaml)
fi

mkdir -p "$(dirname "$ENV_PATH")" "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME"

if [[ -d "$ENV_PATH" ]]; then
  echo "Conda env already exists at '$ENV_PATH'."
else
  "${CREATE_CMD[@]}"
fi

nounset_was_on=0
case "$-" in
  *u*) nounset_was_on=1; set +u ;;
esac
conda activate "$ENV_PATH"
if [[ "$nounset_was_on" -eq 1 ]]; then
  set -u
fi

python -m pip install --upgrade "pip<24.1" "setuptools<81" wheel
python -m pip install --no-build-isolation -r requirements.pace-h200.txt

python - <<'PY'
import h5py
import numpy
import torch
import pytorch_lightning as pl
print("numpy", numpy.__version__)
print("h5py", h5py.__version__)
print("torch", torch.__version__)
print("pytorch_lightning", pl.__version__)
print("cuda available", torch.cuda.is_available())
PY
