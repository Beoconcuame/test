import torch
import torch.nn as nn
import torch.nn.functional as F

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_filters=100, 
                 filter_sizes=[2, 3, 4], num_classes=2, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(fs, embed_dim))
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(f"Input tensor must be 2D, got {x.dim()}D with shape {x.shape}")

        batch_size, seq_len = x.shape
        embeds = self.embedding(x)
        x = embeds.unsqueeze(1)
        conv_outs = [F.relu(conv(x)).squeeze(3) for conv in self.convs]
        pooled_outs = [F.max_pool1d(item, kernel_size=item.size(2)).squeeze(2) for item in conv_outs]
        x = torch.cat(pooled_outs, dim=1)
        x = self.dropout(x)
        logits = self.fc(x)
        return logits
