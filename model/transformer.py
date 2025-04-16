import torch
import torch.nn as nn
import math

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=4,
                 num_classes=2, dropout=0.1, max_len=5000):
        super(TransformerClassifier, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.d_model = embed_dim  

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=max_len+1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dropout=self.dropout_rate,
            activation='gelu',
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.layer_norm = nn.LayerNorm(self.d_model)
        self.bn = nn.BatchNorm1d(self.d_model)

        self.fc = nn.Linear(self.d_model, num_classes)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(f"Input tensor must be 2D, got {x.dim()}D with shape {x.shape}")
        
        embeds = self.embedding(x)


        token_mask = (x != 0)
        key_padding_mask = ~token_mask

        embeds = embeds.transpose(0, 1)

        batch_size = embeds.size(1)

        cls_tokens = self.cls_token.expand(1, batch_size, self.d_model)
        embeds = torch.cat([cls_tokens, embeds], dim=0)

        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=key_padding_mask.device)
        key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

        embeds = self.pos_encoder(embeds)
        transformer_out = self.transformer(embeds, src_key_padding_mask=key_padding_mask)

        cls_output = transformer_out[0]

        out = self.layer_norm(cls_output)
        out = self.bn(out)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return x
