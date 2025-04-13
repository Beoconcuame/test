import torch
import torch.nn as nn

class BiGRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=3,
                 num_classes=2, dropout=0.0, num_features=1):

        super(BiGRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.num_features = num_features
        self.dropout = nn.Dropout(dropout)
        
        input_size = embed_dim if num_features == 1 else embed_dim * num_features
        
        self.bigru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, x):
        if x.dim() == 2:
            if self.num_features > 1:
                x = x.unsqueeze(-1).repeat(1, 1, self.num_features)
        elif x.dim() == 3:
            if self.num_features == 1:
                x = x[:, :, 0]
        else:
            raise ValueError(f"Input tensor must be 2D or 3D, but got {x.dim()}D with shape {x.shape}")
        
        if self.num_features == 1:
            embeds = self.embedding(x) 
        else:
 
            embeds = [self.embedding(x[:, :, i]) for i in range(self.num_features)]
            embeds = torch.cat(embeds, dim=2) 
        
        gru_out, _ = self.bigru(embeds)  
        pooled = torch.mean(gru_out, dim=1)  
        out = self.dropout(pooled)
        logits = self.fc(out)
        return logits
