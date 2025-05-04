import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
import json
import os
from datetime import datetime

# === 1. Data Loading (UNCHANGED) ===
filepath = r'C:\Users\prith\Documents\OpenBCI_GUI\Recordings\OpenBCISession_2025-04-22_12-01-12/BrainFlow-RAW_2025-04-22_12-01-12_0.csv'
columns = ["timestamp", "n1p", "n2p", "n3p", "n4p", "n5p", "n6p", "n7p", "n8p",
          "accel_x", "accel_y", "accel_z", "analog_0", "analog_1", "analog_2",
          "marker", "unused1", "unused2", "unused3", "unused4", "unused5", "unused6", 
          "timestamp_unix", "unused7"]
raw_df = pd.read_csv(filepath, delimiter="\t", names=columns, skiprows=1)

# === 2. Preprocessing (UNCHANGED) ===
channels = ['n1p', 'n2p', 'n3p', 'n4p', 'n5p', 'n6p', 'n7p', 'n8p']
sfreq = 250

def bandpass(data, low=1, high=30, fs=250, order=5):
    nyq = 0.5 * fs
    lowcut = low / nyq
    highcut = high / nyq
    b, a = butter(order, [lowcut, highcut], btype='band')
    return filtfilt(b, a, data)

filtered = {}
for ch in channels:
    filtered[ch] = bandpass(raw_df[ch].values, fs=sfreq)
    filtered[ch] = (filtered[ch] - filtered[ch].mean()) / filtered[ch].std()

# === 3. Epoch Extraction (UNCHANGED) ===
stim_onsets_sec = [5, 9, 13, 17]
stim_onsets_samples = [int(t * sfreq) for t in stim_onsets_sec]
pre_time, post_time = 0.2, 0.6
samples_pre, samples_post = int(pre_time * sfreq), int(post_time * sfreq)
epoch_len = samples_pre + samples_post

stim_features = []
for onset in stim_onsets_samples:
    marker_val = raw_df['marker'].iloc[onset]
    stim_features.append([marker_val])

X_stim = np.array(stim_features, dtype=np.float32)
y = np.zeros((len(stim_onsets_samples), epoch_len, len(channels)), dtype=np.float32)
for ep_idx, onset in enumerate(stim_onsets_samples):
    for ch_idx, ch in enumerate(channels):
        segment = filtered[ch][onset-samples_pre : onset+samples_post]
        if len(segment) == epoch_len:
            y[ep_idx, :, ch_idx] = segment

X_stim_tensor = torch.tensor(X_stim, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# === 4. Model Architecture (FIXED) ===
class StimulusToBrainwave(nn.Module):
    def __init__(self, stimulus_dim=1, hidden_size=64, output_timesteps=200, n_channels=8):
        super().__init__()
        self.stim_encoder = nn.Sequential(
            nn.Linear(stimulus_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64))
        
        self.conv1 = nn.Conv1d(64, 16, kernel_size=3, padding=1)  # Fixed: 64 input channels
        self.lstm = nn.LSTM(16, hidden_size, batch_first=True)
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, n_channels * output_timesteps),
            nn.Unflatten(1, (output_timesteps, n_channels)))
    
    def forward(self, x):
        encoded = self.stim_encoder(x).unsqueeze(-1)  # (batch, 64, 1)
        print(f"Encoded shape: {encoded.shape}")
        cnn_out = self.conv1(encoded)                # (batch, 16, 1)
        print(f"CNN output shape: {cnn_out.shape}")
        cnn_out = cnn_out.permute(0, 2, 1)          # (batch, 1, 16)
        print(f"Permuted shape: {cnn_out.shape}")
        lstm_out, _ = self.lstm(cnn_out)            # (batch, 1, hidden_size)
        print(f"LSTM output shape: {lstm_out.shape}")
        return self.decoder(lstm_out[:, -1, :])      # (batch, 200, 8)

# Initialize
model = StimulusToBrainwave(
    stimulus_dim=X_stim.shape[1],
    output_timesteps=epoch_len,
    n_channels=len(channels)
)

# === 5. Training Setup (UNCHANGED) ===
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def train_model(model, X, y, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

# === 6. Evaluation (UNCHANGED) ===
def evaluate_model(model, X, y):
    model.eval()
    with torch.no_grad():
        preds = model(X)
        mse = criterion(preds, y).item()
        ss_res = torch.sum((y - preds)**2)
        ss_tot = torch.sum((y - y.mean())**2)
        r2 = 1 - ss_res/ss_tot
    return mse, r2.item()

# === 7. Save Function (UNCHANGED) ===
def save_model(model, metrics, metadata, save_dir='saved_models'):
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_dir, f'stim2eeg_{timestamp}.pth')
    torch.save(model.state_dict(), model_path)
    metadata.update({
        'saved_at': timestamp,
        'metrics': {
            'mse': metrics[0],
            'r2': metrics[1]
        }
    })
    with open(os.path.join(save_dir, f'metadata_{timestamp}.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

# === 8. Run Pipeline ===
if __name__ == "__main__":
    train_model(model, X_stim_tensor, y_tensor, epochs=100)
    mse, r2 = evaluate_model(model, X_stim_tensor, y_tensor)
    print(f"\nFinal Metrics:\nMSE: {mse:.4f}\nR²: {r2:.4f}")
    metadata = {
        'channels': channels,
        'sampling_rate': sfreq,
        'stimulus_dim': X_stim.shape[1],
        'model_architecture': str(model)
    }
    save_model(model, (mse, r2), metadata)

###use the following to load model
#model = StimulusToBrainwave()  # With same init parameters
#model.load_state_dict(torch.load('saved_models/stim2eeg_[timestamp].pth'))