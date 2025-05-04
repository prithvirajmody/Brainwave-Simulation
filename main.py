import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# === Step 1: Load OpenBCI CSV ===
filepath = r'C:\Users\prith\Documents\OpenBCI_GUI\Recordings\OpenBCISession_2025-04-22_12-01-12/BrainFlow-RAW_2025-04-22_12-01-12_0.csv'
#raw_df = pd.read_csv(filepath, skiprows=5)
#raw_df = pd.read_csv(filepath, delimiter='\t')

columns = [
    "timestamp", "n1p", "n2p", "n3p", "n4p", "n5p", "n6p", "n7p", "n8p",
    "accel_x", "accel_y", "accel_z",
    "analog_0", "analog_1", "analog_2",
    "marker", "unused1", "unused2", "unused3", "unused4", "unused5", "unused6", "timestamp_unix", "unused7"
]

raw_df = pd.read_csv(filepath, delimiter="\t", names=columns, skiprows=1)


print(raw_df.columns.tolist())

# === Step 2: Define electrode channels you used ===
# Update this list based on the channels you actually connected (Cyton pins: n1p, n2p...)
channels = ['n1p', 'n2p', 'n3p', 'n4p', 'n5p', 'n6p', 'n7p', 'n8p']

# === Step 3: Preprocess EEG ===
sfreq = 250  # Cyton default sample rate (change if yours was different)

def bandpass(data, low=1, high=30, fs=250, order=5):
    nyq = 0.5 * fs
    lowcut = low / nyq
    highcut = high / nyq
    b, a = butter(order, [lowcut, highcut], btype='band')
    return filtfilt(b, a, data)

# Apply filter
filtered = {}
for ch in channels:
    filtered[ch] = bandpass(raw_df[ch].values, fs=sfreq)

# === Step 4: Manually define event onsets (in seconds) ===
# Replace with actual PsychoPy timestamps if available
# Example: Stimuli shown at 5s, 9s, 13s...
stim_onsets_sec = [5, 9, 13, 17]
stim_onsets_samples = [int(t * sfreq) for t in stim_onsets_sec]

# === Step 5: Epoch the signal ===
pre_time = 0.2  # 200ms before stimulus
post_time = 0.6  # 600ms after stimulus

samples_pre = int(pre_time * sfreq)
samples_post = int(post_time * sfreq)
epoch_len = samples_pre + samples_post

epochs = {ch: [] for ch in channels}

for onset in stim_onsets_samples:
    for ch in channels:
        segment = filtered[ch][onset - samples_pre : onset + samples_post]
        if len(segment) == epoch_len:
            epochs[ch].append(segment)

# === Step 6: Average across trials and plot ERP ===
time_axis = np.linspace(-pre_time, post_time, epoch_len)

plt.figure(figsize=(12, 6))
for ch in channels:
    erp_avg = np.mean(epochs[ch], axis=0)
    plt.plot(time_axis * 1000, erp_avg, label=ch)

plt.axvline(0, color='black', linestyle='--', label='Stimulus Onset')
plt.title('ERP Waveform (P1/N1 Candidate Components)')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (μV)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
