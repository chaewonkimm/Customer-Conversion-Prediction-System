import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attention_fc = nn.Linear(hidden_dim, 1)  # 어텐션 가중치를 계산하기 위한 선형 레이어
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # lstm_out: [batch_size, seq_len, hidden_dim]
        
        # 로컬 어텐션 메커니즘
        attn_weights = torch.tanh(self.attention_fc(lstm_out))  # [batch_size, seq_len, 1]
        attn_weights = torch.softmax(attn_weights, dim=1)  # [batch_size, seq_len, 1]
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)  # [batch_size, hidden_dim]
        
        # Fully Connected Layers
        x = F.relu(self.fc1(context_vector))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

def init_model(input_dim, hidden_dim, output_dim, num_layers, dropout):
    return LSTMModel(input_dim, hidden_dim, output_dim, num_layers, dropout)
