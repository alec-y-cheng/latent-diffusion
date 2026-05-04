import pandas as pd
import os
import sys

csv_path = "logs/2026-04-29T23-23-43_uk_pinns_baseline/epoch_metrics.csv"
if not os.path.exists(csv_path):
    print("CSV not found.")
    sys.exit(0)

# The CSV has a header with 14 columns, but the data rows have 33 columns.
# We will read it line by line, find the maximum number of columns, and infer the header
# if it was written alphabetically. Actually, the easiest way is to read the latest metrics.csv
# or just recreate a dummy header.
lines = open(csv_path).readlines()
data = []
for line in lines[1:]: # Skip broken header
    row = line.strip().split(',')
    if len(row) > 1:
        data.append(row)

# The last row has all the values. We know the first two columns are epoch and step.
# Wait, let's just let the pandas logger rewrite the file cleanly from now on.
# But for the current file, we can just leave it or rename it so pandas starts fresh.
# We'll rename it to epoch_metrics_corrupted.csv and let the new pandas logger start a new epoch_metrics.csv.
os.rename(csv_path, csv_path.replace(".csv", "_corrupted.csv"))
print("Renamed corrupted CSV.")
