import torch
import torch.nn as nn

class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=3, 
                 num_classes=2, dropout=0.5, num_features=1):
        super(BiLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_features = num_features
        self.dropout = nn.Dropout(dropout)
        
        self.input_size = embed_dim if num_features == 1 else embed_dim * num_features

        self.bilstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0
        )

        self.fc = nn.Linear(self.hidden_dim * 2, self.num_classes)
    
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
        pooled = torch.mean(lstm_out, dim=1)
        out = self.dropout(pooled)
        logits = self.fc(out)
        return logits
