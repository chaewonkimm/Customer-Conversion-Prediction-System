import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.local_attention_fc = nn.Linear(hidden_dim * 2, 1)  # Bi-directional GRU이므로 hidden_dim * 2
        self.global_attention_fc = nn.Linear(hidden_dim * 2, 1, bias=False)
        self.fc1 = nn.Linear(hidden_dim * 4, hidden_dim)  # local과 global context를 합치므로 hidden_dim * 4
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_second, x_minute, x_hour):
        # 각 입력을 GRU에 통과시킴
        gru_out_second, _ = self.gru(x_second)  # [batch_size, seq_len, hidden_dim * 2]
        gru_out_minute, _ = self.gru(x_minute)  # [batch_size, seq_len, hidden_dim * 2]
        gru_out_hour, _ = self.gru(x_hour)  # [batch_size, seq_len, hidden_dim * 2]
        
        # 모든 GRU 출력을 결합
        gru_out = torch.cat((gru_out_second, gru_out_minute, gru_out_hour), dim=1)  # [batch_size, seq_len * 3, hidden_dim * 2]
        
        # 로컬 어텐션 메커니즘
        local_scores = torch.tanh(self.local_attention_fc(gru_out))  # [batch_size, seq_len * 3, 1]
        local_weights = torch.softmax(local_scores, dim=1)  # [batch_size, seq_len * 3, 1]
        local_context_vector = torch.sum(local_weights * gru_out, dim=1)  # [batch_size, hidden_dim * 2]
        
        # 글로벌 어텐션 메커니즘
        global_scores = self.global_attention_fc(gru_out)  # [batch_size, seq_len * 3, 1]
        global_weights = torch.softmax(global_scores, dim=1)  # [batch_size, seq_len * 3, 1]
        global_context_vector = torch.sum(global_weights * gru_out, dim=1)  # [batch_size, hidden_dim * 2]
        
        # 로컬과 글로벌 컨텍스트 벡터 결합
        combined_context_vector = torch.cat((local_context_vector, global_context_vector), dim=1)  # [batch_size, hidden_dim * 4]
        
        # Fully Connected Layers
        x = F.relu(self.fc1(combined_context_vector))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

def init_model(input_dim, hidden_dim, output_dim, num_layers, dropout):
    return GRUModel(input_dim, hidden_dim, output_dim, num_layers, dropout)