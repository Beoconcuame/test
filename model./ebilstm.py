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
                 num_classes=2, dropout=0.5, num_features=1, pretrained_embeddings=None):
        super(EnhancedBiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if pretrained_embeddings is not None:
            self.embedding.weight = nn.Parameter(pretrained_embeddings, requires_grad=True)
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_features = num_features
        self.dropout_rate = dropout
        self.input_size = embed_dim if num_features == 1 else embed_dim * num_features
        
        self.bilstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0
        )
        
        self.attention = Attention(hidden_dim)
        
        fc_input_dim = self.hidden_dim * 2 * 3
        self.fc = nn.Linear(fc_input_dim, self.num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        if x.dim() == 2 and self.num_features > 1:
            x = x.unsqueeze(-1).repeat(1, 1, self.num_features)
        elif x.dim() == 3 and self.num_features == 1:
            x = x[:, :, 0]
        
        if self.num_features == 1:
            embeds = self.embedding(x)
        else:
            embeds = [self.embedding(x[:, :, i]) for i in range(self.num_features)]
            embeds = torch.cat(embeds, dim=2)
        
        lstm_out, _ = self.bilstm(embeds)
        attn_output = self.attention(lstm_out)
        mean_output = torch.mean(lstm_out, dim=1)
        max_output, _ = torch.max(lstm_out, dim=1)
        
        combined = torch.cat([attn_output, mean_output, max_output], dim=1)
        out = self.dropout(combined)
        logits = self.fc(out)
        return logits
