import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]

class TransformerWithCombinedPositionalEncoding(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        custom_pos_vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float = 0.1,
        max_len: int = 5000,
        combine_mode: str = "sum",
        pos_weight: float = 0.2,
    ):
        super().__init__()
        mode = combine_mode.lower()
        if mode not in ("sum", "concat"):
            raise ValueError("combine_mode must be 'sum' or 'concat'")
        self.combine_mode = mode
        self.pos_weight = pos_weight
        self.text_weight = 1.0 - pos_weight

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.custom_pos_embedding = nn.Embedding(custom_pos_vocab_size, d_model)
        self.sinusoidal_encoding = SinusoidalPositionalEncoding(d_model, max_len)

        if mode == "concat":
            self.proj = nn.Linear(d_model * 2, d_model)

        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        token_ids: torch.Tensor,
        custom_positions: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        token_embeds = self.token_embedding(token_ids)
        standard_pos = self.sinusoidal_encoding(token_embeds)
        custom_pos = self.custom_pos_embedding(custom_positions)

        if self.combine_mode == "sum":
            combined_pos = standard_pos + custom_pos
            combined = self.text_weight * token_embeds + self.pos_weight * combined_pos
        else:
            base = self.text_weight * token_embeds + self.pos_weight * standard_pos
            combined = torch.cat([base, custom_pos], dim=-1)
            combined = self.proj(combined)

        combined = self.dropout(combined)
        x = self.transformer_encoder(combined, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        logits = self.fc_out(x)
        return logits
