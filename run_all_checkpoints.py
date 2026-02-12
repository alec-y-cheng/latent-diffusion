
import os
import glob
import argparse
import subprocess
import sys

# Experiment list derived from submit_all_experiments.sh
EXPERIMENTS = [
    "medlr_highb_medaux",
    "highlr_highb_medaux",
    "lowlr_highb_medaux",
    "medlr_lowb_medaux",
    "highlr_lowb_medaux",
    "lowlr_lowb_medaux",
    "medlr_lowb_highweight",
    "medlr_lowb_highaux",
    "medlr_lowb_lowaux",
    "medlr_lowb_noaux",
    # Specific run requested by user
    "2026-02-02T01-45-25_cfd_ldm"
]

def find_experiment_artifacts(log_root, exp_name):
    """
    Finds the latest checkpoint and config for a given experiment name.
    """
    # 1. Try exact path match first
    exact_path = os.path.join(log_root, exp_name)
    if os.path.isdir(exact_path):
         latest_dir = exact_path
    else:
        # 2. Pattern matches: timestamp_str_exp_name
        search_pattern = os.path.join(log_root, f"*_{exp_name}")
        dirs = glob.glob(search_pattern)
        
        if not dirs:
            return None, None
            
        # Sort by creation time (or name if timestamp is sortable)
        # Timestamps are YYYY-MM-DDTHH-MM-SS, so string sort works generally
        dirs.sort(reverse=True)
        latest_dir = dirs[0]
    
    # Checkpoint
    ckpt_dir = os.path.join(latest_dir, "checkpoints")
    ckpts = glob.glob(os.path.join(ckpt_dir, "*.ckpt"))
    if not ckpts:
        return None, None
        
    # Prefer 'last.ckpt' if exists, else take latest
    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    if last_ckpt in ckpts:
        target_ckpt = last_ckpt
    else:
        # Sort checks
        ckpts.sort() 
        target_ckpt = ckpts[-1]
        
    # Config
    # main.py saves to configs/*-project.yaml
    cfg_dir = os.path.join(latest_dir, "configs")
    configs = glob.glob(os.path.join(cfg_dir, "*-project.yaml"))
    if not configs:
        # Fallback just in case
        configs = glob.glob(os.path.join(cfg_dir, "*.yaml"))
        
    if not configs:
        return target_ckpt, None
        
    target_config = configs[0] # Take the first one (usually only one project config)
    
    return target_ckpt, target_config

def main():
    parser = argparse.ArgumentParser(description="Run inference_test.py on all experiments.")
    parser.add_argument("--logdir", type=str, default="logs", help="Root directory of logs")
    parser.add_argument("--outdir", type=str, default="inference_results_all", help="Output directory for all results")
    parser.add_argument("--steps", type=int, default=50, help="DDIM steps")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples per model")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.logdir):
        print(f"Error: Log directory '{args.logdir}' not found.")
        return

    os.makedirs(args.outdir, exist_ok=True)
    
    print(f"Scanning {args.logdir} for {len(EXPERIMENTS)} experiments...")
    
    for exp in EXPERIMENTS:
        print(f"\n--- Processing {exp} ---")
        ckpt, config = find_experiment_artifacts(args.logdir, exp)
        
        if not ckpt:
            print(f"  [Skipping] No checkpoints found for {exp}")
            continue
            
        if not config:
            print(f"  [Skipping] No config found for {exp} (found ckpt: {ckpt})")
            continue
            
        print(f"  Found checkpoint: {ckpt}")
        print(f"  Found config:     {config}")
        
        # Create specific output subdir
        exp_outdir = os.path.join(args.outdir, exp)
        
        cmd = [
            "python", "inference_test.py",
            "--config", config,
            "--ckpt", ckpt,
            "--outdir", exp_outdir,
            "--steps", str(args.steps),
            "--num_samples", str(args.num_samples)
        ]
        
        print(f"  Running inference...")
        if args.dry_run:
            print("  [DRY RUN] Command:", " ".join(cmd))
        else:
            try:
                subprocess.run(cmd, check=True)
                print(f"  [Success] Results saved to {exp_outdir}")
            except subprocess.CalledProcessError as e:
                print(f"  [Failed] Error code {e.returncode}")
            except Exception as e:
                print(f"  [Error] {e}")

if __name__ == "__main__":
    main()
