import torch
import torch.nn as nn

class IpaBiLSTMClassifierWithPos(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        custom_pos_vocab_size: int,
        embedding_dim: int = 300,
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout: float = 0.5,
        combine_mode: str = "sum",
        text_weight: float = 0.8,
        pos_weight: float = 0.2,
    ):
        super().__init__()
        mode = combine_mode.lower()
        if mode not in {"sum", "concat"}:
            raise ValueError("combine_mode must be 'sum' or 'concat'")
        self.combine_mode = mode
        self.text_weight = text_weight
        self.pos_weight = pos_weight

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(custom_pos_vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)

        lstm_input_dim = embedding_dim * 2 if self.combine_mode == "concat" else embedding_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, tokens: torch.Tensor, custom_positions: torch.Tensor) -> torch.Tensor:
        token_embeds = self.embedding(tokens)
        pos_embeds = self.pos_embedding(custom_positions)


        if self.combine_mode == "concat":
            combined = torch.cat([token_embeds, pos_embeds], dim=-1)
        else:
            combined = token_embeds * self.text_weight + pos_embeds * self.pos_weight


        combined = self.dropout(combined)

        outputs, (hn, cn) = self.lstm(combined)


        h_fwd = hn[-2]
        h_bwd = hn[-1]
        h = torch.cat([h_fwd, h_bwd], dim=1)
        h = self.dropout(h)


        logits = self.fc(h)
        return logits
