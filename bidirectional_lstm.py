import torch
import torch.nn as nn
import torch.optim as optim

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0, bidirectional=True, bias=True):
        super(BiLSTM, self).__init__()
        
        # LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_size,     # <<< Adjust input size here (number of input features per timestep)
            hidden_size=hidden_size,   # <<< Adjust hidden units here
            num_layers=num_layers,     # <<< Adjust number of stacked LSTM layers here
            dropout=dropout,            # <<< Adjust dropout between LSTM layers here
            bidirectional=bidirectional,  # <<< Set True/False for bidirectional here
            bias=bias,                 # <<< Enable/disable bias terms here
            batch_first=True           # Make input/output tensors have shape (batch, seq, feature)
        )
        
        # Output Layer
        self.fc = nn.Linear(
            hidden_size * (2 if bidirectional else 1),  # multiply by 2 if bidirectional
            output_size
        )
    
    def forward(self, x, hidden=None):
        """
        Forward pass through BiLSTM
        Args:
            x: Tensor of shape (batch_size, seq_len, input_size)
            hidden: Optional initial hidden/cell states
        """
        if hidden is not None:
            # <<< Optionally pass manually initialized hidden states
            lstm_out, hidden = self.lstm(x, hidden)
        else:
            lstm_out, hidden = self.lstm(x)

        # Optionally: sequence packing can be handled here if you use variable length sequences
        # <<< Add sequence packing/unpacking if needed

        out = self.fc(lstm_out)  # Apply FC to each timestep output
        return out, hidden

    def init_hidden(self, batch_size, device):
        """
        Initialize hidden and cell states manually
        """
        num_directions = 2 if self.lstm.bidirectional else 1
        h0 = torch.zeros(self.lstm.num_layers * num_directions, batch_size, self.lstm.hidden_size).to(device)
        c0 = torch.zeros(self.lstm.num_layers * num_directions, batch_size, self.lstm.hidden_size).to(device)
        return (h0, c0)

# Example of using the model
if __name__ == "__main__":
    # --- Adjustable Parameters ---
    input_size = 64      # <<< Change based on your ERP input features
    hidden_size = 128    # <<< Change hidden units
    num_layers = 2       # <<< Change number of layers
    output_size = 64     # <<< Change output size (match input if reconstructing)
    dropout = 0.1        # <<< Change dropout rate
    bidirectional = True # <<< True for BiLSTM
    bias = True          # <<< True to add bias terms

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BiLSTM(input_size, hidden_size, num_layers, output_size, dropout, bidirectional, bias).to(device)
    
    # Example input (batch_size=32, sequence_length=100, input_size=64)
    dummy_input = torch.randn(32, 100, input_size).to(device)
    
    # Initialize hidden states manually (optional)
    hidden = model.init_hidden(batch_size=32, device=device)

    # Forward pass
    output, hidden = model(dummy_input, hidden)

    # --- Optimizer Example ---
    learning_rate = 0.001  # <<< Adjust learning rate here
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # <<< Choose optimizer here

    print("Output shape:", output.shape)  # Should be (batch_size, seq_len, output_size)
