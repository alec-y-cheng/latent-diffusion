
import torch
import glob
import os
import yaml

def inspect_ckpt(path):
    print(f"--- Inspecting {path} ---")
    try:
        sd = torch.load(path, map_location="cpu")
        state_dict = sd.get("state_dict", sd)
        
        # Check for scale_factor in buffer
        if "model.scale_factor" in state_dict:
            print(f"Scale Factor (buffer): {state_dict['model.scale_factor']}")
        else:
            print("Scale Factor (buffer): Not found (likely 1.0 or not scale_by_std)")

        # Check for hyperparameters in Lightning checkpoint
        if "hyper_parameters" in sd:
            print("Hyperparameters found in checkpoint:")
            hparams = sd["hyper_parameters"]
            # recursive print limited
            print(f"  original_elbo_weight: {hparams.get('original_elbo_weight', 'N/A')}")
            print(f"  scale_factor argument: {hparams.get('scale_factor', 'N/A')}")
        else:
            print("No 'hyper_parameters' key in checkpoint.")

        # Try to find config.yaml in the same log directory
        log_dir = os.path.dirname(os.path.dirname(path))
        config_path = os.path.join(log_dir, "configs", "*-project.yaml")
        configs = glob.glob(config_path)
        if not configs:
            configs = glob.glob(os.path.join(log_dir, "configs", "*.yaml"))
            
        if configs:
            print(f"Found config file: {configs[0]}")
            with open(configs[0], 'r') as f:
                conf = yaml.safe_load(f)
                try:
                    params = conf['model']['params']
                    print(f"  Config original_elbo_weight: {params.get('original_elbo_weight', 'Default')}")
                    print(f"  Config scale_factor: {params.get('scale_factor', 'Default')}")
                except:
                    print("  Could not parse model params from config.")
        else:
            print("No config file found in logs.")

    except Exception as e:
        print(f"Error loading {path}: {e}")

# Find checkpoints
base_ckpts = glob.glob("logs/*base*/**/last.ckpt", recursive=True)
med_ckpts = glob.glob("logs/*medlr*/**/last.ckpt", recursive=True)
all_ckpts = base_ckpts + med_ckpts

if not all_ckpts:
    print("No 'last.ckpt' files found in logs/. Checking for any .ckpt")
    all_ckpts = glob.glob("logs/**/*.ckpt", recursive=True)[:5]

for ckpt in all_ckpts:
    inspect_ckpt(ckpt)
