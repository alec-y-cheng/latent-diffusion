import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

def get_experiment_data(logs_dir):
    """
    Scans logs directory for metrics.csv files.
    Groups them by Experiment Name.
    Returns a dict: { 'exp_name': pd.DataFrame (concatenated & sorted) }
    """
    experiments = {}
    
    # scan logs/EXP_NAME/testtube/version_*/metrics.csv
    # adjust wildcard based on actual structure. 
    # Usually: logs/TIMESTAMP_NAME/testtube/version_0/metrics.csv
    
    print(f"Scanning {logs_dir}...")
    
    # Iterate over timestamped folders
    for folder in os.listdir(logs_dir):
        path = os.path.join(logs_dir, folder)
        if not os.path.isdir(path): continue
        
        # Skip autoencoders
        if "autoencoder" in folder.lower(): continue
        
        # Parse Experiment Name
        # 2026-02-18T..._name_of_exp
        parts = folder.split('_')
        if len(parts) < 2: continue
        exp_name = "_".join(parts[1:])
        
        # Find metrics.csv inside testtube/version_*
        # Note: Sometimes it's testtube/version_0/metrics.csv
        # or just testtube/metrics.csv depending on Logger setup
        
        metrics_files = glob.glob(os.path.join(path, "testtube", "version_*", "metrics.csv"))
        
        # Also check directly in folder or other patterns if needed
        if not metrics_files:
             metrics_files = glob.glob(os.path.join(path, "testtube", "metrics.csv"))
             
        if not metrics_files:
            continue
            
        if exp_name not in experiments:
            experiments[exp_name] = []
            
        for mf in metrics_files:
            try:
                df = pd.read_csv(mf)
                experiments[exp_name].append(df)
            except Exception as e:
                print(f"Error reading {mf}: {e}")

    # Aggregation
    aggregated = {}
    
    for exp_name, dfs in experiments.items():
        if not dfs: continue
        
        # Concatenate all runs for this experiment
        full_df = pd.concat(dfs, ignore_index=True)
        
        # Filter for validation loss
        # Key might be 'val/loss_simple', 'val/loss', etc.
        # Let's check columns
        
        # Relevant columns: epoch, step, val/loss_simple (or similar)
        # We want to plot Val Loss vs Epoch
        
        # Sort by epoch/step
        if 'epoch' in full_df.columns:
            full_df = full_df.sort_values('epoch')
        elif 'step' in full_df.columns:
            full_df = full_df.sort_values('step')
            
        aggregated[exp_name] = full_df
        
    return aggregated

def plot_losses(data, output_path):
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    
    # Metric to plot
    # Try looking for 'val/loss_simple', 'val/loss', 'val/rec_loss'
    metric_keys = ['val/loss_simple', 'val/loss', 'val/rec_loss']
    
    plotted_count = 0
    
    for exp_name, df in data.items():
        # Find available metric
        y_key = None
        for k in metric_keys:
            if k in df.columns:
                # Check if it has non-nan values
                if df[k].notna().sum() > 0:
                    y_key = k
                    break
        
        if not y_key:
            print(f"Skipping {exp_name}: No validation loss found in columns {df.columns}")
            continue
            
        # Clean NaNs (Validation often runs less frequently than training)
        viz_df = df.dropna(subset=[y_key])
        
        if len(viz_df) == 0:
            continue
            
        # Smooth the curve
        # Rolling mean to reduce noise
        if len(viz_df) > 10:
            window = 10
            viz_df[y_key] = viz_df[y_key].rolling(window=window, min_periods=1, center=True).mean()
            
            # Subsample (plot every Nth point) after smoothing
            viz_df = viz_df.iloc[::5] # Plot every 5th point for cleaner graph
            
        # Plot
        # Use 'epoch' if available, else 'step' or index
        if 'epoch' in viz_df.columns:
            x_axis = viz_df['epoch']
            x_label = "Epoch"
        elif 'step' in viz_df.columns:
            x_axis = viz_df['step']
            x_label = "Step"
        else:
            x_axis = range(len(viz_df))
            x_label = "Log Interval"
            
        # Smooth curve slightly?
        plt.plot(x_axis, viz_df[y_key], label=f"{exp_name}", alpha=0.9, linewidth=1)
        plotted_count += 1
        
    if plotted_count == 0:
        print("Nothing to plot!")
        return

    plt.title("Validation Loss Comparison", fontsize=16)
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Log scale y-axis often helps if losses vary strictly
    plt.yscale('log')
    
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=str, default="logs")
    parser.add_argument("--out", type=str, default="results/validation_loss_comparison.png")
    args = parser.parse_args()
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    data = get_experiment_data(args.logs)
    print(f"Found {len(data)} experiments with data.")
    
    plot_losses(data, args.out)

if __name__ == "__main__":
    main()
