import torch
import torch.nn as nn

class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=3,
                 num_classes=2, dropout=0.0):
        super(BiGRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        
        self.bigru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(f"Input tensor must be 2D, but got {x.dim()}D with shape {x.shape}")
        
        embeds = self.embedding(x)
        gru_out, _ = self.bigru(embeds)
        pooled = torch.mean(gru_out, dim=1)
        out = self.dropout(pooled)
        logits = self.fc(out)
        return logits
