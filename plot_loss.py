
import pandas as pd
import matplotlib.pyplot as plt
import os

# File path
csv_path = r"/Users/dcheng/latent-diffusion/metrics/base.csv"
output_path = r"/Users/dcheng/latent-diffusion/metrics/base.png"

# Load data
try:
    df = pd.read_csv(csv_path)
    print("Data loaded successfully.")
    
    # Check for NaNs
    if df.isnull().values.any():
        print("Warning: NaNs found in data (this is normal for epoch-based metrics in step-based logs).")
        # We should fill NaNs or drop rows where the metric doesn't exist?
        # Actually, let's just use forward fill or dropna for plotting specific columns
    
    # print(f"Stats:\n{df.describe()}")

except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# plotting
plt.figure(figsize=(12, 6))

# Plot Training Loss
# We filter to ignore step-based NaNs if any
train_df = df.dropna(subset=['train/loss_epoch'])
plt.plot(train_df['epoch'], train_df['train/loss_epoch'], label='Train Loss (Epoch)', linewidth=1.5)

# Plot Validation Loss
# Check if val/loss exists (it usually logs less frequently)
if 'val/loss' in df.columns:
    val_df = df.dropna(subset=['val/loss'])
    plt.plot(val_df['epoch'], val_df['val/loss'], label='Validation Loss', alpha=0.7, linewidth=1.5, marker='o')

plt.xlabel('Epoch')
plt.ylabel('Loss (Log Scale)')
plt.title('Training and Validation Loss History')
plt.legend()
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.yscale('log') # Log scale is often better for loss

# Save plot
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
