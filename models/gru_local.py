import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention_fc = nn.Linear(hidden_dim * 2, 1)  # Bi-directional GRU이므로 hidden_dim * 2
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        gru_out, _ = self.gru(x)  # gru_out: [batch_size, seq_len, hidden_dim * 2]
        
        # 로컬 어텐션 메커니즘
        attn_scores = torch.tanh(self.attention_fc(gru_out))  # [batch_size, seq_len, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch_size, seq_len, 1]
        context_vector = torch.sum(attn_weights * gru_out, dim=1)  # [batch_size, hidden_dim * 2]
        
        # Fully Connected Layers
        x = F.relu(self.fc1(context_vector))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

def init_model(input_dim, hidden_dim, output_dim, num_layers, dropout):
    return GRUModel(input_dim, hidden_dim, output_dim, num_layers, dropout)