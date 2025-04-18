import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=3,
                 num_classes=2, dropout=0.5, padding_idx=0):
        super(BiLSTMClassifier, self).__init__()
        self.num_classes = num_classes 
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx) 
        self.dropout = nn.Dropout(dropout)

        self.bilstm = nn.LSTM(
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
        embeds = self.dropout(embeds) 
        lstm_out, _ = self.bilstm(embeds)
        pooled = torch.mean(lstm_out, dim=1)
        pooled_dropout = self.dropout(pooled)


        logits = self.fc(pooled_dropout)
        return logits
