import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.context_vector = nn.Parameter(torch.randn(hidden_dim * 2))
    
    def forward(self, lstm_out):
        energy = torch.tanh(self.attn(lstm_out))
        attention_weights = torch.matmul(energy, self.context_vector)
        attention_weights = F.softmax(attention_weights, dim=1)
        attended_output = torch.sum(lstm_out * attention_weights.unsqueeze(-1), dim=1)
        return attended_output

class EnhancedBiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=3, 
                 num_classes=2, dropout=0.5, pretrained_embeddings=None):
        super(EnhancedBiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings, requires_grad=True)
        
        self.dropout = nn.Dropout(dropout)
        
        self.bilstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.attention = Attention(hidden_dim)
        

        fc_input_dim = hidden_dim * 2 * 3  
        self.fc = nn.Linear(fc_input_dim, num_classes)
    
    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(f"Input tensor must be 2D, but got {x.dim()}D with shape {x.shape}")
        
        embeds = self.embedding(x)
        lstm_out, _ = self.bilstm(embeds)
        attn_output = self.attention(lstm_out)
        mean_output = torch.mean(lstm_out, dim=1)
        max_output, _ = torch.max(lstm_out, dim=1)
        
        combined = torch.cat([attn_output, mean_output, max_output], dim=1)
        out = self.dropout(combined)
        logits = self.fc(out)
        return logits
