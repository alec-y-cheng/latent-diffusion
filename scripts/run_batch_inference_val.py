import os
import argparse
import glob
import subprocess
from datetime import datetime

def get_experiment_groups(logs_dir):
    """
    Groups log directories by experiment name.
    """
    groups = {}
    
    if not os.path.exists(logs_dir):
        print(f"Error: Logs directory '{logs_dir}' not found.")
        return {}

    for folder in os.listdir(logs_dir):
        path = os.path.join(logs_dir, folder)
        if not os.path.isdir(path):
            continue
            
        # Parse timestamp: YYYY-MM-DDTHH-MM-SS_name
        try:
            # 2026-02-02T01-45-25_cfd_ldm
            parts = folder.split('_')
            if len(parts) < 2:
                continue
                
            timestamp_str = parts[0]
            exp_name = "_".join(parts[1:])
            
            # Verify timestamp format
            dt = datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S")
            
            if exp_name not in groups:
                groups[exp_name] = []
            
            groups[exp_name].append({
                'path': path,
                'timestamp': dt,
                'folder': folder
            })
            
        except ValueError:
            continue

    return groups

def get_best_checkpoint(folder):
    """
    Finds the best checkpoint in a log folder.
    Prioritizes: last.ckpt -> best.ckpt -> epoch=*.ckpt (latest mtime)
    """
    ckpt_dir = os.path.join(folder, "checkpoints")
    if not os.path.exists(ckpt_dir):
        return None
        
    # 1. last.ckpt (Often most relevant for resuming, but for inference maybe 'best'?)
    # User said "latest best checkpoints".
    # Usually 'last.ckpt' is the very latest state.
    last = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.exists(last):
        return last
        
    # 2. *.ckpt
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        return None
        
    # Sort by mtime
    ckpts.sort(key=os.path.getmtime, reverse=True)
    return ckpts[0]

def get_config_path(folder):
    """
    Finds project.yaml or configs/*.yaml
    """
    # 1. Standard location copies
    cfg = os.path.join(folder, "configs", "project.yaml")
    if os.path.exists(cfg):
        return cfg
        
    # 2. Any yaml in configs/
    cfg_dir = os.path.join(folder, "configs")
    if os.path.exists(cfg_dir):
        yamls = glob.glob(os.path.join(cfg_dir, "*.yaml"))
        if yamls:
            return yamls[0]

    return None

def main():
    parser = argparse.ArgumentParser(description="Run inference on validation set for latest checkpoints.")
    parser.add_argument("--logs", type=str, default="logs", help="Path to logs directory")
    parser.add_argument("--outdir_suffix", type=str, default="val_inference_results", help="Dir name inside log folder")
    parser.add_argument("--default_config", type=str, default="configs/latent-diffusion/cfd_ldm.yaml", help="Fallback config")
    parser.add_argument("--data_path", type=str, default=None, help="Override validation data path")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to generate")
    parser.add_argument("--steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--filter", type=str, default=None, help="Filter experiments by name (partial match)")
    args = parser.parse_args()
    
    # 1. Locate script
    script_path = os.path.join(os.path.dirname(__file__), "inference_test.py")
    if not os.path.exists(script_path):
        # Fallback to current dir if running from scripts/
        if os.path.exists("inference_test.py"):
            script_path = "inference_test.py"
        else:
            print("Error: Could not find scripts/inference_test.py")
            return

    # 2. Scan Groups
    print(f"Scanning {args.logs}...")
    groups = get_experiment_groups(args.logs)
    
    if not groups:
        print("No valid experiment folders found.")
        return

    print(f"Found {len(groups)} experiments.")
    
    for exp_name, runs in groups.items():
        if args.filter and args.filter not in exp_name:
            continue
            
        # Sort desc
        runs.sort(key=lambda x: x['timestamp'], reverse=True)
        latest_run = runs[0]
        
        print(f"\n--- {exp_name} ---")
        print(f"Latest Run: {latest_run['folder']}")
        
        # Checkpoint
        ckpt = get_best_checkpoint(latest_run['path'])
        if not ckpt:
            print("  [Skipping] No checkpoints found.")
            continue
            
        # Config
        config = get_config_path(latest_run['path'])
        if not config:
            print(f"  [Warning] No config in logs. Using default: {args.default_config}")
            config = args.default_config
            
        # Output Dir
        outdir = os.path.join(latest_run['path'], args.outdir_suffix)
        
        # Build Command
        cmd = [
            "python", script_path,
            "--config", config,
            "--ckpt", ckpt,
            "--outdir", outdir,
            "--num_samples", str(args.num_samples),
            "--steps", str(args.steps)
        ]
        
        if args.data_path:
            cmd.extend(["--data_path", args.data_path])
            
        print(f"  Running inference...")
        try:
            # Run and pipe output so we see progress
            subprocess.run(cmd, check=True)
            print(f"  [Success] Saved to {outdir}")
        except subprocess.CalledProcessError as e:
            print(f"  [Failed] Exit code {e.returncode}")

    # 3. Aggregate All Results
    print("\n--- Aggregating All Results ---")
    all_summaries = []
    
    for exp_name, runs in groups.items():
        # Re-find the path we just used/would have used
        runs.sort(key=lambda x: x['timestamp'], reverse=True)
        latest_run = runs[0]
        outdir = os.path.join(latest_run['path'], args.outdir_suffix)
        summary_csv = os.path.join(outdir, "summary_metrics.csv")
        
        if os.path.exists(summary_csv):
            try:
                import pandas as pd
                df = pd.read_csv(summary_csv, index_col=0)
                # Usually row 0 is mean, row 1 is std.
                # Let's extract Mean row and add Experiment Name
                mean_row = df.loc['mean'].copy()
                mean_row['Experiment'] = exp_name
                mean_row['Timestamp'] = latest_run['timestamp']
                all_summaries.append(mean_row)
            except Exception as e:
                print(f"  Error reading {summary_csv}: {e}")
    
    if all_summaries:
        import pandas as pd
        master_df = pd.DataFrame(all_summaries)
        # Reorder so Experiment is first
        cols = ['Experiment', 'Timestamp'] + [c for c in master_df.columns if c not in ['Experiment', 'Timestamp']]
        master_df = master_df[cols]
        
        master_csv = "all_experiments_summary.csv"
        master_df.to_csv(master_csv, index=False)
        print(f"Saved master summary to: {os.path.abspath(master_csv)}")
        print(master_df)
    else:
        print("No summary metrics found to aggregate.")

if __name__ == "__main__":
    main()
