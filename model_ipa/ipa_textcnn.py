import torch
import torch.nn as nn
import torch.nn.functional as F

class IpaTextCNNClassifierWithPos(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        custom_pos_vocab_size: int,
        embedding_dim: int = 300,
        num_classes: int = 2,
        filter_sizes: tuple[int, ...] = (3, 4, 5),
        num_filters: int = 100,
        dropout: float = 0.5,
        combine_mode: str = "sum",
        text_weight: float = 0.8,
        pos_weight: float = 0.2,
    ):
        super().__init__()
        mode = combine_mode.lower()
        if mode not in ("sum", "concat"):
            raise ValueError("combine_mode must be 'sum' or 'concat'")
        self.combine_mode = mode
        self.text_weight = text_weight
        self.pos_weight = pos_weight

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(custom_pos_vocab_size, embedding_dim)

        conv_input_dim = embedding_dim * 2 if mode == "concat" else embedding_dim
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, conv_input_dim)) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, tokens: torch.Tensor, custom_positions: torch.Tensor) -> torch.Tensor:
        token_embeds = self.embedding(tokens)
        pos_embeds = self.pos_embedding(custom_positions)

        if self.combine_mode == "concat":
            x = torch.cat((token_embeds, pos_embeds), dim=-1)
        else:
            x = token_embeds * self.text_weight + pos_embeds * self.pos_weight

        x = x.unsqueeze(1)
        conv_outs = []
        for conv in self.convs:
            conv_x = F.relu(conv(x))
            conv_x = conv_x.squeeze(3)
            pooled = F.max_pool1d(conv_x, conv_x.size(2))
            conv_outs.append(pooled.squeeze(2))

        out = torch.cat(conv_outs, dim=1)
        out = self.dropout(out)
        logits = self.fc(out)
        return logits
