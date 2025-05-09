import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error
import optuna
import os
import json
from datetime import datetime

# ==========================
# Load Data
# ==========================
train_tensor = torch.load("train_epochs.pt")
test_tensor = torch.load("test_epochs.pt")

# Print data information
print(f"Train tensor shape: {train_tensor.shape}")
print(f"Test tensor shape: {test_tensor.shape}")
print(f"Train tensor stats - min: {train_tensor.min()}, max: {train_tensor.max()}, mean: {train_tensor.mean()}")

# Data normalization - normalize the data to prevent numerical instabilities
def normalize_data(data):
    # Scale the data to [-1, 1] range
    data = torch.Tensor(data)
    data_min = torch.min(data)
    data_max = torch.max(data)
    normalized = 2 * (data - data_min) / (data_max - data_min) - 1
    return normalized, data_min, data_max

# Normalize training and test data
train_tensor_norm, train_min, train_max = normalize_data(train_tensor)
# Use same normalization parameters for test data to ensure consistency
test_tensor_norm = 2 * (torch.Tensor(test_tensor) - train_min) / (train_max - train_min) - 1

print(f"Normalized train tensor stats - min: {train_tensor_norm.min()}, max: {train_tensor_norm.max()}, mean: {train_tensor_norm.mean()}")

# ==========================
# Model Definitions
# ==========================
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, p_drop=0.1):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),  # LeakyReLU for more stable gradients
            nn.BatchNorm1d(hidden_dim),  # Add BatchNorm for stability
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),  # Add BatchNorm for stability
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p_drop),
            nn.Linear(hidden_dim, input_dim)
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        # Clamp logvar to prevent extreme values
        logvar = torch.clamp(self.fc_logvar(h), min=-10, max=10)
        return mu, logvar

    def reparameterize(self, mu, logvar, temperature=1.0):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + temperature * eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, temperature=1.0):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar, temperature)
        recon = self.decode(z)
        return recon, mu, logvar, z

class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_size, latent_dim, num_layers=1, dropout=0.1, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, dropout=dropout if num_layers > 1 else 0,
                            batch_first=True, bidirectional=bidirectional)
        multiplier = 2 if bidirectional else 1
        self.fc = nn.Linear(hidden_size * multiplier, latent_dim)

    def forward(self, x):
        # Handle zero-length sequences
        if x.size(1) == 0:
            batch_size = x.size(0)
            multiplier = 2 if self.lstm.bidirectional else 1
            return torch.zeros(batch_size, self.fc.out_features, device=x.device)
            
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        return self.fc(last_step)

# ==========================
# Loss Functions
# ==========================
def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    # Handle potential NaN values in inputs
    if torch.isnan(recon_x).any() or torch.isnan(x).any() or torch.isnan(mu).any() or torch.isnan(logvar).any():
        print("Warning: NaN values detected in VAE loss inputs")
        
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='mean')
    # KL divergence with numerical stability
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Check for NaN in loss components
    if torch.isnan(recon_loss).any():
        print("Warning: NaN values in reconstruction loss")
        recon_loss = torch.tensor(1.0, device=recon_x.device, requires_grad=True)
    
    if torch.isnan(kl).any():
        print("Warning: NaN values in KL divergence")
        kl = torch.tensor(0.1, device=recon_x.device, requires_grad=True)
        
    return recon_loss + beta * kl, recon_loss.item(), kl.item()

def latent_loss(pred_z, true_z):
    # Check for NaN values
    if torch.isnan(pred_z).any() or torch.isnan(true_z).any():
        print("Warning: NaN values detected in latent loss inputs")
        
    return nn.functional.mse_loss(pred_z, true_z, reduction='mean')

# ==========================
# Training Functions
# ==========================
def check_and_report_nan(tensor, name):
    """Utility function to check for NaN values"""
    if torch.isnan(tensor).any():
        print(f"Warning: NaN values detected in {name}")
        return True
    return False

def train_epoch_vae(vae, dataloader, optimizer, beta, temperature, clip_value=1.0):
    vae.train()
    total_loss = total_recon = total_kl = 0
    num_batches = 0
    
    for x in dataloader:
        x = x.float()
        # Check for NaN in input
        if check_and_report_nan(x, "VAE input"):
            continue
            
        x_flat = x.view(-1, x.shape[-1])
        recon, mu, logvar, _ = vae(x_flat)
        
        # Check for NaN in model outputs
        if (check_and_report_nan(recon, "VAE recon") or 
            check_and_report_nan(mu, "VAE mu") or 
            check_and_report_nan(logvar, "VAE logvar")):
            continue
            
        loss, recon_l, kl_l = vae_loss(recon, x_flat, mu, logvar, beta)
        
        # Check for NaN in loss
        if check_and_report_nan(loss, "VAE loss"):
            continue
            
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(vae.parameters(), clip_value)
        
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_l
        total_kl += kl_l
        num_batches += 1
        
    if num_batches == 0:
        return float('nan'), float('nan'), float('nan')
        
    return total_loss / num_batches, total_recon / num_batches, total_kl / num_batches

def train_epoch_bilstm(bilstm, vae, dataloader, optimizer, clip_value=1.0):
    bilstm.train()
    vae.eval()
    total_loss = 0
    num_batches = 0
    
    for x in dataloader:
        x = x.float()
        # Check for NaN in input
        if check_and_report_nan(x, "BiLSTM input"):
            continue
            
        x_flat = x.view(-1, x.shape[-1])
        
        try:
            with torch.no_grad():
                _, _, _, z = vae(x_flat)
                z = z.view(x.shape[0], x.shape[1], -1).mean(dim=1)
                
                # Check for NaN in VAE output
                if check_and_report_nan(z, "VAE latent output"):
                    continue
                
            pred_z = bilstm(x)
            
            # Check for NaN in BiLSTM output
            if check_and_report_nan(pred_z, "BiLSTM output"):
                continue
                
            loss = latent_loss(pred_z, z)
            
            # Check for NaN in loss
            if check_and_report_nan(loss, "BiLSTM loss"):
                continue
                
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(bilstm.parameters(), clip_value)
            
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            
        except RuntimeError as e:
            print(f"RuntimeError in BiLSTM training: {e}")
            continue
    
    if num_batches == 0:
        return float('nan')
        
    return total_loss / num_batches

def evaluate_bilstm(bilstm, vae, dataloader):
    bilstm.eval()
    vae.eval()
    all_true = []
    all_pred = []
    
    with torch.no_grad():
        for x in dataloader:
            x = x.float()
            
            # Skip batches with NaN
            if check_and_report_nan(x, "Eval input"):
                continue
                
            x_flat = x.view(-1, x.shape[-1])
            
            try:
                _, _, _, z = vae(x_flat)
                z = z.view(x.shape[0], x.shape[1], -1).mean(dim=1)
                
                # Skip batches with NaN VAE output
                if check_and_report_nan(z, "Eval VAE output"):
                    continue
                    
                pred_z = bilstm(x)
                
                # Skip batches with NaN BiLSTM output
                if check_and_report_nan(pred_z, "Eval BiLSTM output"):
                    continue
                    
                all_true.append(z)
                all_pred.append(pred_z)
                
            except RuntimeError as e:
                print(f"RuntimeError in evaluation: {e}")
                continue
    
    if not all_true or not all_pred:
        print("Warning: No valid predictions for evaluation")
        return float('inf'), float('inf')
        
    true_all = torch.cat(all_true).cpu().numpy()
    pred_all = torch.cat(all_pred).cpu().numpy()
    
    # Final check for NaN values before metric calculation
    if np.isnan(true_all).any() or np.isnan(pred_all).any():
        print("Warning: NaN values still present in final evaluation arrays")
        # Replace NaN with zeros
        true_all = np.nan_to_num(true_all)
        pred_all = np.nan_to_num(pred_all)
    
    mse = mean_squared_error(true_all, pred_all)
    rmse = np.sqrt(mse)
    return mse, rmse

# ==========================
# Optuna Objective Function
# ==========================
def objective(trial):
    # Print trial start
    print(f"Starting Trial {trial.number}")

    # Define hyperparameter search space - slightly reduced to focus on stability
    vae_hidden_dim = trial.suggest_categorical("vae_hidden_dim", [64, 128])
    vae_latent_dim = trial.suggest_categorical("vae_latent_dim", [16, 32])
    vae_p_drop = trial.suggest_float("vae_p_drop", 0.1, 0.3)
    vae_lr = trial.suggest_float("vae_lr", 1e-4, 5e-3, log=True)  # Reduced upper bound
    vae_beta = trial.suggest_float("vae_beta", 0.5, 2.0)  # Reduced range
    vae_temperature = trial.suggest_float("vae_temperature", 0.8, 1.2)  # More conservative
    
    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [32, 64])
    lstm_num_layers = trial.suggest_int("lstm_num_layers", 1, 2)  # Reduced to avoid overfitting
    lstm_dropout = trial.suggest_float("lstm_dropout", 0.1, 0.3)
    lstm_lr = trial.suggest_float("lstm_lr", 1e-4, 5e-3, log=True)  # Reduced upper bound
    batch_size = trial.suggest_categorical("batch_size", [16, 32])  # Smaller batch sizes for stability

    # Update DataLoader with new batch size
    train_loader = DataLoader(train_tensor_norm, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_tensor_norm, batch_size=batch_size, shuffle=False)

    # Initialize models
    vae = VAE(INPUT_DIM, vae_hidden_dim, vae_latent_dim, p_drop=vae_p_drop)
    bilstm = BiLSTM(INPUT_DIM, lstm_hidden_size, vae_latent_dim, num_layers=lstm_num_layers, dropout=lstm_dropout)

    # Use weight decay to prevent exploding weights
    optimizer_vae = optim.Adam(vae.parameters(), lr=vae_lr, weight_decay=1e-5)
    optimizer_bilstm = optim.Adam(bilstm.parameters(), lr=lstm_lr, weight_decay=1e-5)

    # Learning rate schedulers for more stable training
    vae_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_vae, 'min', factor=0.5, patience=3)
    bilstm_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer_bilstm, 'min', factor=0.5, patience=3)

    # Train VAE with early stopping
    EPOCHS = 20
    best_vae_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        total, recon, kl = train_epoch_vae(vae, train_loader, optimizer_vae, vae_beta, vae_temperature)
        
        # Skip NaN loss epochs
        if np.isnan(total):
            print(f"[VAE] Trial {trial.number}, Epoch {epoch+1}/{EPOCHS}: NaN loss detected, skipping")
            return float('inf')  # Skip this trial
            
        # Convert metrics to Python float for JSON serialization
        trial.set_user_attr(f"vae_epoch_{epoch}_total_loss", float(total))
        trial.set_user_attr(f"vae_epoch_{epoch}_recon_loss", float(recon))
        trial.set_user_attr(f"vae_epoch_{epoch}_kl_div", float(kl))
        
        # Update learning rate
        vae_scheduler.step(total)
        
        # Early stopping check
        if total < best_vae_loss:
            best_vae_loss = total
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"[VAE] Early stopping at epoch {epoch+1}")
            break
            
        # Print epoch progress
        print(f"[VAE] Trial {trial.number}, Epoch {epoch+1}/{EPOCHS}, Loss: {total:.6f}")

    # Train BiLSTM with early stopping
    best_bilstm_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(EPOCHS):
        loss = train_epoch_bilstm(bilstm, vae, train_loader, optimizer_bilstm)
        
        # Skip NaN loss epochs
        if np.isnan(loss):
            print(f"[BiLSTM] Trial {trial.number}, Epoch {epoch+1}/{EPOCHS}: NaN loss detected, skipping")
            return float('inf')  # Skip this trial
            
        # Convert metric to Python float
        trial.set_user_attr(f"bilstm_epoch_{epoch}_latent_loss", float(loss))
        
        # Update learning rate
        bilstm_scheduler.step(loss)
        
        # Early stopping check
        if loss < best_bilstm_loss:
            best_bilstm_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f"[BiLSTM] Early stopping at epoch {epoch+1}")
            break
            
        # Print epoch progress
        print(f"[BiLSTM] Trial {trial.number}, Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.6f}")

    # Evaluate on test set with error handling
    try:
        test_mse, test_rmse = evaluate_bilstm(bilstm, vae, test_loader)
        
        if np.isnan(test_rmse) or np.isinf(test_rmse):
            print(f"Trial {trial.number}: Invalid RMSE value")
            return float('inf')
            
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return float('inf')

    # Log final metrics as Python float
    trial.set_user_attr("test_mse", float(test_mse))
    trial.set_user_attr("test_rmse", float(test_rmse))
    
    print(f"Trial {trial.number} completed with RMSE: {test_rmse:.6f}")

    # Save models if this is the best trial so far
    global best_rmse
    if best_rmse is None or test_rmse < best_rmse:
        best_rmse = test_rmse
        # Save VAE and BiLSTM models
        torch.save(vae.state_dict(), os.path.join(save_dir, f"vae_best_trial_{trial.number}.pth"))
        torch.save(bilstm.state_dict(), os.path.join(save_dir, f"bilstm_best_trial_{trial.number}.pth"))
        # Save hyperparameters and metrics
        metadata = {
            "trial_number": trial.number,
            "rmse": float(test_rmse),  # Ensure float
            "mse": float(test_mse),    # Ensure float
            "params": trial.params,
            "metrics": {k: float(v) if isinstance(v, (np.float32, np.float64)) else v for k, v in trial.user_attrs.items()}
        }
        with open(os.path.join(save_dir, f"best_trial_{trial.number}_metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)

    return test_rmse

# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    # Setup logging
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    # Setup save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"model_checkpoints_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Initialize best RMSE tracker
    best_rmse = None

    # Define constants
    INPUT_DIM = train_tensor.shape[2]
    print(f"Input dimension: {INPUT_DIM}")

    # Create Optuna study with more robust pruner
    study = optuna.create_study(
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=42)  # Add seed for reproducibility
    )
    
    # Add a catch handler for interrupted trials
    try:
        study.optimize(objective, n_trials=20, catch=(ValueError, RuntimeError))
    except KeyboardInterrupt:
        print("Optimization interrupted by user.")

    # Print best trial results
    if study.best_trial:
        print("\nBest trial:")
        trial = study.best_trial
        print(f"  RMSE: {trial.value}")
        print("  Parameters: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print("  Additional metrics: ")
        for key, value in trial.user_attrs.items():
            print(f"    {key}: {value}")
        print(f"  Best models saved in: {save_dir}/vae_best_trial_{trial.number}.pth and bilstm_best_trial_{trial.number}.pth")
        print(f"  Metadata saved in: {save_dir}/best_trial_{trial.number}_metadata.json")
    else:
        print("No successful trials were completed.")

    # Save the study for later analysis
    study_save_path = os.path.join(save_dir, "study.pkl")
    joblib.dump(study, study_save_path)
    print(f"Study saved to: {study_save_path}")

    # Optionally, visualize results if visualization packages are available
    try:
        import matplotlib.pyplot as plt
        
        # Plot optimization history
        plt.figure(figsize=(10, 6))
        optuna.visualization.matplotlib.plot_optimization_history(study)
        plt.savefig(os.path.join(save_dir, "optimization_history.png"))
        
        # Plot parameter importances if there are enough trials
        if len(study.trials) > 5:
            plt.figure(figsize=(10, 6))
            optuna.visualization.matplotlib.plot_param_importances(study)
            plt.savefig(os.path.join(save_dir, "param_importances.png"))
            
        print(f"Visualization plots saved to {save_dir}")
    except (ImportError, ValueError) as e:
        print(f"Could not generate visualizations: {e}")