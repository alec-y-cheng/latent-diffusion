
import pandas as pd
import matplotlib.pyplot as plt
import os

# File path
print(os.getcwd())

csv_path = "./metrics_full.csv"
output_path = "./loss_plot_full.png"

# Load data
try:
    df = pd.read_csv(csv_path)
    print("Data loaded successfully.")
    
    # Check for NaNs
    if df.isnull().values.any():
        print("Warning: NaNs found in data.")
    
    print(f"Stats:\n{df.describe()}")

except Exception as e:
    print(f"Error loading CSV: {e}")
    exit(1)

# plotting
plt.figure(figsize=(12, 6))

# Filter NaNs separately because train and val are logged at different steps
train_data = df.dropna(subset=['train/loss_simple_epoch'])
val_data = df.dropna(subset=['val/loss'])

plt.plot(train_data['epoch'], train_data['train/loss_simple_epoch'], label='Train Loss', linewidth=1.5)
plt.plot(val_data['epoch'], val_data['val/loss'], label='Validation Loss', alpha=0.7, linewidth=1.5, marker='o') # Add marker to see sparse val points
plt.xlabel('Epoch')
plt.ylabel('Loss (Log scale)')
plt.title('Training and Validation Loss History')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.yscale('log')

# Save plot
plt.savefig(output_path)
print(f"Plot saved to {output_path}")
