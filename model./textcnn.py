import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=100, 
                 filter_sizes=[2, 3, 4], num_classes=2, dropout=0.5, num_features=1):

        super(TextCNN, self).__init__()
        self.embed_dim = embed_dim
        self.num_features = num_features
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.effective_embed_dim = embed_dim * num_features
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, self.effective_embed_dim))
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        if x.dim() == 2:
            batch_size, seq_len = x.shape
            x = x.unsqueeze(2)
        elif x.dim() == 3:
            batch_size, seq_len, features = x.shape
            if features != self.num_features:
                raise ValueError(f"Input features ({features}) do not match expected num_features ({self.num_features}).")
        else:
            raise ValueError(f"Input tensor must be 2D or 3D, got {x.dim()}D.")
        
        embeds = self.embedding(x)
        embeds = embeds.view(batch_size, seq_len, self.effective_embed_dim)
        x = embeds.unsqueeze(1)
        conv_outs = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled_outs = [F.max_pool1d(item, kernel_size=item.size(2)).squeeze(2) for item in conv_outs]
        x = torch.cat(pooled_outs, dim=1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
