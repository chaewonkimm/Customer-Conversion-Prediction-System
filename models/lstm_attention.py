import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=num_heads, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        residual = attn_output[:, -1, :]
        x = F.relu(self.fc1(residual))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

def init_model(input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout):
    return LSTMAttentionModel(input_dim, hidden_dim, output_dim, num_layers, num_heads, dropout)

