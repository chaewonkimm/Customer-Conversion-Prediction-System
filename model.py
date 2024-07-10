import torch
import torch.nn as nn
import torch.nn.functional as F

## Hierarchical Architecture
class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.local_attention_fc = nn.Linear(hidden_dim * 2, 1)
        self.global_attention_fc = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_second, x_minute, x_hour):
        gru_out_second, _ = self.gru(x_second)
        gru_out_minute, _ = self.gru(x_minute)
        gru_out_hour, _ = self.gru(x_hour)
        
        gru_out = torch.cat((gru_out_second, gru_out_minute, gru_out_hour), dim=1)
        
        ## Local Attention Mechanism
        local_scores = torch.tanh(self.local_attention_fc(gru_out))
        local_weights = torch.softmax(local_scores, dim=1)
        local_context_vector = torch.sum(local_weights * gru_out, dim=1)
        
        ## Global Attention Mechanism
        global_scores = self.global_attention_fc(gru_out)
        global_weights = torch.softmax(global_scores, dim=1)
        global_context_vector = torch.sum(global_weights * gru_out, dim=1)
        
        combined_context_vector = torch.cat((local_context_vector, global_context_vector), dim=1)
        
        x = F.relu(self.fc1(combined_context_vector))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

def init_model(input_dim, hidden_dim, output_dim, num_layers, dropout):
    return GRUModel(input_dim, hidden_dim, output_dim, num_layers, dropout)