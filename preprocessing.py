import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import torch
from sklearn.model_selection import train_test_split

# ----------------------
# Configuration Settings
# ----------------------
SAMPLING_RATE = 250  # Hz
LOWCUT = 1.0          # Hz
HIGHCUT = 30.0        # Hz
FILTER_ORDER = 4
PRE_STIMULUS_MS = 200  # milliseconds before marker
POST_STIMULUS_MS = 800 # milliseconds after marker
ARTIFACT_THRESHOLD = 100  # microvolts

# ----------------------
# Bandpass Filter Function
# ----------------------
def bandpass_filter(data, lowcut=LOWCUT, highcut=HIGHCUT, fs=SAMPLING_RATE, order=FILTER_ORDER):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data, axis=0)

# ----------------------
# Main Preprocessing Function
# ----------------------
def preprocess_eeg(file_path):
    # Load TSV file
    df = pd.read_csv(file_path, sep='\t')

    # Extract EEG channels and markers
    eeg = df[['n1p', 'n2p', 'n3p', 'n4p', 'n5p', 'n6p', 'n7p', 'n8p']].values
    markers = df['marker'].values

    # Apply bandpass filter
    eeg = bandpass_filter(eeg)

    # Find event indices (non-zero markers)
    event_indices = np.where(markers != 0)[0]

    # Convert ms to samples
    pre_stim = int(PRE_STIMULUS_MS / 1000 * SAMPLING_RATE)
    post_stim = int(POST_STIMULUS_MS / 1000 * SAMPLING_RATE)
    epoch_len = pre_stim + post_stim

    # Create epochs
    epochs = []
    for idx in event_indices:
        if idx - pre_stim >= 0 and idx + post_stim <= len(eeg):
            epoch = eeg[idx - pre_stim : idx + post_stim]
            epochs.append(epoch)

    epochs = np.stack(epochs)  # (n_epochs, epoch_len, n_channels)

    # Baseline correction (subtract pre-stimulus mean)
    baseline_corrected = epochs - np.mean(epochs[:, :pre_stim, :], axis=1, keepdims=True)

    # Artefact rejection
    clean_epochs = baseline_corrected[np.max(np.abs(baseline_corrected), axis=(1, 2)) < ARTIFACT_THRESHOLD]

    # Convert to PyTorch tensor
    X = torch.tensor(clean_epochs, dtype=torch.float32)  # Shape: (n_epochs, epoch_len, n_channels)

    # Save tensor
    torch.save(epochs, "tensor.pt")

    #LOAD TENSOR WITH COMMAND epochs = torch.load("training_tensor.pt")

    return X

# ----------------------
# Example Usage
# ----------------------
if __name__ == '__main__':
    file_path = 'filtered_readings.tsv'
    data_tensor = preprocess_eeg(file_path)
    print(f"Preprocessed data shape: {data_tensor.shape}")
