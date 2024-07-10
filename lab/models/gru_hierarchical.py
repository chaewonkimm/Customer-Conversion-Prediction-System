import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUAttentionLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.attention = nn.Linear(hidden_dim * 2, 1, bias=False)
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        attn_scores = torch.tanh(self.attention(gru_out))
        attn_weights = torch.softmax(attn_scores, dim=1)
        context_vector = torch.sum(attn_weights * gru_out, dim=1)
        return context_vector

class HierarchicalAttentionGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # GRU and Attention layers for seconds, minutes, and hours
        self.gru_attention_second = GRUAttentionLayer(input_dim, hidden_dim, num_layers, dropout)
        self.gru_attention_minute = GRUAttentionLayer(hidden_dim * 2, hidden_dim, num_layers, dropout)
        self.gru_attention_hour = GRUAttentionLayer(hidden_dim * 2, hidden_dim, num_layers, dropout)
        
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_second, x_minute, x_hour):
        # Process second-level sequences
        context_vector_second = self.gru_attention_second(x_second)
        
        # Process minute-level sequences
        context_vector_minute = self.gru_attention_minute(context_vector_second.unsqueeze(1))
        
        # Process hour-level sequences
        context_vector_hour = self.gru_attention_hour(context_vector_minute.unsqueeze(1))
        
        x = F.relu(self.fc1(context_vector_hour))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

def init_model(input_dim, hidden_dim, output_dim, num_layers, dropout):
    return HierarchicalAttentionGRUModel(input_dim, hidden_dim, output_dim, num_layers, dropout)