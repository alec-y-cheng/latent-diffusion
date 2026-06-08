#!/bin/bash

activate_ldm_env() {
  local env_name="${LDM_ENV_NAME:-ldm}"
  local env_path="${LDM_ENV_PATH:-/storage/scratch1/7/acheng324/conda/envs/$env_name}"
  export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/storage/scratch1/7/acheng324/conda/pkgs}"
  export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/storage/scratch1/7/acheng324/.cache}"
  export CONDA_NO_PLUGINS="${CONDA_NO_PLUGINS:-true}"

  module load anaconda3/2023.03

  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available after loading anaconda3/2023.03." >&2
    echo "Check available modules with: module avail anaconda" >&2
    exit 2
  fi

  eval "$(conda shell.bash hook)"
  mkdir -p "$(dirname "$env_path")" "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME"

  if [[ ! -d "$env_path" ]]; then
    echo "ERROR: conda environment '$env_name' does not exist on this cluster." >&2
    echo "Create it first, for example:" >&2
    echo "  bash scripts/create_ldm_env_pace.sh" >&2
    echo "Expected env location: $env_path" >&2
    echo "Or set LDM_ENV_NAME to an existing environment." >&2
    echo "Or set LDM_ENV_PATH to an existing conda environment prefix." >&2
    exit 2
  fi

  local nounset_was_on=0
  case "$-" in
    *u*) nounset_was_on=1; set +u ;;
  esac
  conda activate "$env_path"
  if [[ "$nounset_was_on" -eq 1 ]]; then
    set -u
  fi

  python - <<'PY'
import sys
missing = []
for name in ("torch", "pytorch_lightning", "omegaconf", "h5py"):
    try:
        __import__(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    print("ERROR: Active Python environment is missing required packages:", file=sys.stderr)
    for item in missing:
        print(f"  - {item}", file=sys.stderr)
    sys.exit(2)
PY

  echo "Activated conda env: $CONDA_DEFAULT_ENV"
  echo "Python: $(which python)"
}

stage_file_to_tmp() {
  local src="$1"
  local tmp_root="${LDM_JOB_TMPDIR:-${TMPDIR:-/tmp}}"
  if [[ "$tmp_root" == "/tmp" || "$tmp_root" == "/var/tmp" ]]; then
    tmp_root="/tmp/${USER:-${LOGNAME:-user}}/${SLURM_JOB_ID:-$$}"
  fi

  if [[ ! -f "$src" ]]; then
    echo "ERROR: dataset file not found: $src" >&2
    exit 3
  fi

  local rel="${src#/}"
  local dst="$tmp_root/$rel"
  export LDM_JOB_TMPDIR="$tmp_root"
  export TMPDIR="$tmp_root"
  mkdir -p "$tmp_root"
  mkdir -p "$(dirname "$dst")"

  echo "Copying dataset to local NVMe: $src -> $dst"
  cp "$src" "$dst"
  echo "Copy complete. Size: $(du -h "$dst" | cut -f1)"

  STAGED_DATA_FILE="$dst"
}

stage_dir_to_tmp() {
  local src="$1"
  local tmp_root="${LDM_JOB_TMPDIR:-${TMPDIR:-/tmp}}"
  if [[ "$tmp_root" == "/tmp" || "$tmp_root" == "/var/tmp" ]]; then
    tmp_root="/tmp/${USER:-${LOGNAME:-user}}/${SLURM_JOB_ID:-$$}"
  fi

  if [[ ! -d "$src" ]]; then
    echo "ERROR: dataset directory not found: $src" >&2
    exit 3
  fi

  local rel="${src#/}"
  local dst="$tmp_root/$rel"
  export LDM_JOB_TMPDIR="$tmp_root"
  export TMPDIR="$tmp_root"
  mkdir -p "$tmp_root"
  mkdir -p "$dst"

  echo "Copying dataset directory to local NVMe: $src -> $dst"
  cp -r "$src"/* "$dst"/
  echo "Copy complete."

  STAGED_DATA_DIR="$dst"
}
