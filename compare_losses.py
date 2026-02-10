import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import argparse

def compare_models(folder_path, output_path):
    print(f"Scanning {folder_path} for CSV files...")
    # Find all CSV files in the folder
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not files:
        print("No CSV files found.")
        return

    plt.figure(figsize=(12, 6))
    
    plotted_count = 0
    for file_path in files:
        file_name = os.path.basename(file_path)
        try:
            df = pd.read_csv(file_path)
            
            # Identify the validation loss column
            val_col = None
            if 'val/loss' in df.columns:
                val_col = 'val/loss'
            elif 'val_loss' in df.columns:
                val_col = 'val_loss'
                
            if val_col:
                # Filter NaNs for plotting
                df_clean = df.dropna(subset=[val_col, 'epoch'])
                if not df_clean.empty:
                    plt.plot(df_clean['epoch'], df_clean[val_col], label=f'{file_name}', alpha=0.8, marker='.')
                    plotted_count += 1
            else:
                print(f"Skipping {file_name}: No 'val/loss' column found.")
                
        except Exception as e:
            print(f"Error reading {file_name}: {e}")

    if plotted_count > 0:
        plt.xlabel('Epoch')
        plt.ylabel('Validation Loss (Log Scale)')
        plt.title('Validation Loss Comparison')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.yscale('log')
        
        plt.savefig(output_path)
        print(f"\nComparison plot saved to {output_path}")
    else:
        print("No valid data to plot.")

if __name__ == "__main__":
    # Default Paths
    DEFAULT_FOLDER = "/Users/dcheng/latent-diffusion/results"
    DEFAULT_OUTPUT = "/Users/dcheng/latent-diffusion/results/model_comparison.png"
    
    compare_models(DEFAULT_FOLDER, DEFAULT_OUTPUT)
