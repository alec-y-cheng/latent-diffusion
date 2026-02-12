
import pandas as pd
import glob
import os

files = glob.glob("metrics/*.csv")
results = []

print(f"{'Model':<40} | {'Min Val/Loss':<12} | {'Epoch':<5} | {'Weight (Est)':<10}")
print("-" * 80)

for f in files:
    try:
        df = pd.read_csv(f)
        if 'val/loss' in df.columns:
            min_loss = df['val/loss'].min()
            min_epoch = df.loc[df['val/loss'].idxmin(), 'epoch']
            
            # Estimate weight from filename keywords
            weight = "N/A"
            if "noaux" in f: weight = "0.0"
            elif "medaux" in f: weight = "1e-4"
            elif "highaux" in f: weight = "1.0"
            elif "highweight" in f: weight = "0.1" # from script comments
            elif "base" in f: weight = "0.0 (Def)"
            
            results.append((f, min_loss, min_epoch, weight))
    except Exception as e:
        print(f"Error reading {f}: {e}")

results.sort(key=lambda x: x[1])

for r in results:
    print(f"{os.path.basename(r[0]):<40} | {r[1]:.6f}     | {r[2]:<5} | {r[3]:<10}")
