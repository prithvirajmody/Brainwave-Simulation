import pandas as pd
import numpy as np
import csv

# Load the file
df = pd.read_csv(
    r'C:\Users\prith\Documents\OpenBCI_GUI\Recordings\OpenBCISession_2025-04-22_12-01-12\BrainFlow-RAW_2025-04-22_12-01-12_0.csv',
    header=None
)

# Step 1: Split the tab-separated values
split_data = df[0].str.split("\t", expand=True)

# Step 2: Convert to float
data = split_data.astype(float).values  # shape: (num_samples, num_columns)
print("Raw shape:", data.shape)

# Full column headers from original OpenBCI format
all_columns = [
    "timestamp", "n1p", "n2p", "n3p", "n4p", "n5p", "n6p", "n7p", "n8p",
    "accel_x", "accel_y", "accel_z",
    "analog_0", "analog_1", "analog_2",
    "marker", "unused1", "unused2", "unused3", "unused4", "unused5", "unused6", "timestamp_unix", "unused7"
]

# Columns to keep
selected_columns = ["timestamp", "n1p", "n2p", "n3p", "n4p", "n5p", "n6p", "n7p", "n8p", "marker"]

# Find indices of selected columns
selected_indices = [all_columns.index(col) for col in selected_columns]

# Extract selected columns from the data
filtered_data = data[:, selected_indices]

# Estimate sampling rate (based on timestamp)
time_col = filtered_data[:, 0]
deltas = np.diff(time_col)
mean_delta = np.mean(deltas)
sampling_rate = round(1 / mean_delta)
print(f"Estimated sampling rate: {sampling_rate} Hz")

# Save to TSV
output_path = 'filtered_readings.tsv'

# Write to TSV with headers
with open(output_path, 'w', newline='') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(selected_columns)
    writer.writerows(filtered_data)

print(f"Filtered data saved to: {output_path}")
