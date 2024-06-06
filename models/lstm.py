import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        residual = lstm_out[:, -1, :]
        x = F.relu(self.fc1(residual))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

def init_model(input_dim, hidden_dim, output_dim, num_layers, dropout):
    return LSTMModel(input_dim, hidden_dim, output_dim, num_layers, dropout)
