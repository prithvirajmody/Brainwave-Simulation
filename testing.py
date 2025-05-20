################################################################
#This code is going to use the saved VAE and BiLSTM models and test to see how good their simulations are
#################################################################

#Necessary libraries
import torch  # PyTorch for tensor operations and neural networks
import torch.nn as nn  # Neural network modules
import torch.optim as optim  # Optimizers for training
from torch.utils.data import DataLoader  # For batching and loading data
import numpy as np  # For numerical operations
from sklearn.metrics import mean_squared_error  # For computing MSE and RMSE
import os  # For file and directory operations
import json  # For saving metadata
from datetime import datetime  # For timestamp generation

#Re-defining the VAE and Bi-LSTM model structure
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, p_drop=0.1):
        # Initialize the parent nn.Module class
        super(VAE, self).__init__()
        # Define the encoder: maps input to a hidden representation
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # First linear layer
            nn.LeakyReLU(0.2),  # LeakyReLU activation to allow small negative gradients
            nn.BatchNorm1d(hidden_dim),  # Batch normalization for training stability (addresses internal covariant shift by reducing the dependence of layer inputs on the parameters of earlier layers.)
            nn.Dropout(p_drop),  # Dropout to prevent overfitting
            nn.Linear(hidden_dim, hidden_dim),  # Second linear layer
            nn.LeakyReLU(0.2)  # Another LeakyReLU activation (this helps fight the dying relu issue)
        )
        # Linear layers to compute the mean (mu) and log-variance (logvar) of the latent distribution
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Define the decoder: maps latent space back to input space
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),  # First linear layer
            nn.LeakyReLU(0.2),  # LeakyReLU activation
            nn.BatchNorm1d(hidden_dim),  # Batch normalization
            nn.Linear(hidden_dim, hidden_dim),  # Second linear layer
            nn.LeakyReLU(0.2),  # Another LeakyReLU activation
            nn.Dropout(p_drop),  # Dropout
            nn.Linear(hidden_dim, input_dim)  # Output layer to reconstruct input
        )

    def encode(self, x):
        # Pass input through the encoder to get a hidden representation
        h = self.encoder(x)
        # Compute the mean (mu) of the latent distribution
        mu = self.fc_mu(h)
        # Compute the log-variance (logvar) and clamp to prevent extreme values
        logvar = torch.clamp(self.fc_logvar(h), min=-10, max=10)
        return mu, logvar

    def reparameterize(self, mu, logvar, temperature=1.0):
        # Compute the standard deviation from log-variance
        std = torch.exp(0.5 * logvar)
        # Generate random noise with the same shape as std
        eps = torch.randn_like(std)
        # Apply reparameterization trick: z = mu + std * eps * temperature
        return mu + temperature * eps * std

    def decode(self, z):
        # Pass latent representation through the decoder to reconstruct input
        return self.decoder(z)

    def forward(self, x, temperature=1.0):
        # Encode input to get mean and log-variance
        mu, logvar = self.encode(x)
        # Sample latent vector using reparameterization
        z = self.reparameterize(mu, logvar, temperature)
        # Decode latent vector to reconstruct input
        recon = self.decode(z)
        # Return reconstructed input, mean, log-variance, and latent vector
        return recon, mu, logvar, z

# Define the Bidirectional LSTM (BiLSTM) model
class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim, num_layers=1, dropout=0.1, bidirectional=True):  ##Consider increasing num layers once i dont have to train the model on his weak ahh laptop 
        ##Also increase dropout probability if overfitting becomes a problem
        # Initialize the parent nn.Module class
        super(BiLSTM, self).__init__()
        # Define the LSTM layer with specified parameters
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, dropout=dropout if num_layers > 1 else 0,
                            batch_first=True, bidirectional=bidirectional)
        # Compute multiplier (2 for bidirectional, 1 for unidirectional)
        multiplier = 2 if bidirectional else 1
        # Define a linear layer to map LSTM output to the latent dimension
        self.fc = nn.Linear(hidden_size * multiplier, latent_dim)

    def forward(self, x):
        # Handle edge case of zero-length sequences
        if x.size(1) == 0:
            batch_size = x.size(0)
            multiplier = 2 if self.lstm.bidirectional else 1
            # Return zero tensor with appropriate shape
            return torch.zeros(batch_size, self.fc.out_features, device=x.device)
        # Pass input through the LSTM
        lstm_out, _ = self.lstm(x)
        # Extract the last time step's output
        last_step = lstm_out[:, -1, :]
        # Map to latent dimension using the fully connected layer
        return self.fc(last_step)
    
#Loading model weights for .pth files
vae = VAE(input_dim=8, latent_dim=16, hidden_dim=128)
vae.load_state_dict(torch.load(r"model_checkpoints_20250507_163838\vae_best_trial_1.pth"))
vae.eval()

bilstm = BiLSTM(input_dim=8, hidden_size=64, latent_dim=16, num_layers=2, bidirectional=True)
bilstm.load_state_dict(torch.load(r"model_checkpoints_20250507_163838\bilstm_best_trial_1.pth"))
bilstm.eval()

#Simulating ERP waves (Sample from VAE latent space)
with torch.no_grad():
    z = torch.randn(1, 16)  # sample a random latent vector
    simulated_eeg = vae.decode(z)  # shape: [1, input_dim]
    simulated_eeg = simulated_eeg.unsqueeze(1)  # make it [1, seq_len, features] for BiLSTM
    simulated_erp = bilstm(simulated_eeg)  # shape: [1, seq_len, output_dim]

#Simulating ERP waves (Encode real EEG)
'''
real_eeg = torch.tensor(real_eeg_np, dtype=torch.float32)  # shape [1, input_dim]
with torch.no_grad():
    mu, logvar = vae.encode(real_eeg)
    z = vae.reparameterize(mu, logvar)
    simulated_eeg = vae.decode(z)
    simulated_eeg = simulated_eeg.unsqueeze(1)
    simulated_erp = bilstm(simulated_eeg)
'''

#Plotting simulated waveform
import matplotlib.pyplot as plt

simulated_erp_np = simulated_erp.squeeze().numpy()
plt.plot(simulated_erp_np)
plt.title("Simulated P1/N1 ERP Waveform")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude")
plt.grid(True)
plt.show()
