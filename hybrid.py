import torch
import torch.nn as nn
import torch.optim as optim

# ==========================
# VAE Components
# ==========================

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, p_drop=0.1):
        super(VAE, self).__init__()
        
        # --------- ENCODER ---------
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p_drop),   # <--- Dropout for regularization
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)     # <--- Mean output
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim) # <--- Log variance output
        
        # --------- DECODER ---------
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim, input_dim)   # Output same shape as input
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar, temperature=1.0):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + temperature * eps * std  # Temperature for sampling smoothness

    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x, temperature=1.0):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, temperature)
        recon = self.decode(z)
        return recon, mu, logvar, z

# ==========================
# BiLSTM Component
# ==========================

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim, num_layers=1, dropout=0.1, bidirectional=True):
        super(BiLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
            bias=True
        )
        
        direction_multiplier = 2 if bidirectional else 1
        
        # Fully connected layer to predict latent vector
        self.fc = nn.Linear(hidden_size * direction_multiplier, latent_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # Outputs (batch, seq_len, hidden*directions)
        lstm_out_last = lstm_out[:, -1, :]  # Take last time step output
        latent_pred = self.fc(lstm_out_last)
        return latent_pred

# ==========================
# Loss Functions
# ==========================

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')   # Reconstruction loss
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())  # KL divergence
    return recon_loss + beta * kl_loss

def latent_loss(pred_z, true_z):
    return nn.functional.mse_loss(pred_z, true_z, reduction='mean')   # Latent space MSE loss

# ==========================
# Training Skeleton
# ==========================

# --- Hyperparameters (adjust freely) ---
INPUT_DIM = 64           # Number of ERP features
HIDDEN_DIM = 128         # Hidden layer size for VAE
LATENT_DIM = 32          # Latent vector dimension
LSTM_HIDDEN_SIZE = 64    # LSTM hidden size
NUM_LAYERS = 2           # Number of LSTM layers
P_DROP = 0.2             # Dropout probability
BETA = 4.0               # Beta value for VAE loss
TEMPERATURE = 1.0        # VAE sampling temperature
LR = 1e-3                # Learning rate
BATCH_SIZE = 32          # Batch size

# --- Instantiate models ---
vae = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM, p_drop=P_DROP)
bilstm = BiLSTM(INPUT_DIM, LSTM_HIDDEN_SIZE, LATENT_DIM, num_layers=NUM_LAYERS, dropout=P_DROP)

# --- Optimizers ---
optimizer_vae = optim.Adam(vae.parameters(), lr=LR)
optimizer_bilstm = optim.Adam(bilstm.parameters(), lr=LR)

# ==========================
# Example Forward Pass (Training loop will come next)
# ==========================

# x_batch: Tensor of shape (batch_size, sequence_length, input_dim)
# Example dummy input for checking
x_batch = torch.randn(BATCH_SIZE, 100, INPUT_DIM)  # (batch, seq, features)

# --- VAE pretraining (reconstruction) ---
flattened = x_batch.view(-1, INPUT_DIM)   # Flatten batch and seq to simulate independent ERPs
recon_batch, mu, logvar, z = vae(flattened)

loss_vae = vae_loss(recon_batch, flattened, mu, logvar, beta=BETA)

# --- BiLSTM latent prediction ---
pred_z = bilstm(x_batch)

loss_latent = latent_loss(pred_z, z.detach())  # Detach z from VAE graph

# --- Backpropagation (for each model separately first) ---
optimizer_vae.zero_grad()
loss_vae.backward()
optimizer_vae.step()

optimizer_bilstm.zero_grad()
loss_latent.backward()
optimizer_bilstm.step()

# ==========================
# Full Training loops would separately pretrain VAE, then train BiLSTM.
# ==========================

