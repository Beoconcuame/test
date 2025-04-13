import torch
import torch.nn as nn
import math

class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_heads=4, num_layers=4,
                 num_classes=2, dropout=0.1, num_features=3, max_len=5000):
        super(TransformerClassifier, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout
        self.num_features = num_features
        self.d_model = embed_dim if num_features == 1 else embed_dim * num_features

        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Positional Encoding, mở rộng thêm cho [CLS] token
        self.pos_encoder = PositionalEncoding(self.d_model, max_len=max_len+1)

        # Transformer Encoder Layer với activation GELU và pre-norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dropout=self.dropout_rate,
            activation='gelu',
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        # Layer normalization và batch normalization
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.bn = nn.BatchNorm1d(self.d_model)

        # Fully connected classification head
        self.fc = nn.Linear(self.d_model, num_classes)
        self.dropout = nn.Dropout(self.dropout_rate)

    def forward(self, x):
        # Điều chỉnh kích thước đầu vào nếu cần
        if x.dim() == 2 and self.num_features > 1:
            x = x.unsqueeze(-1).repeat(1, 1, self.num_features)
        elif x.dim() == 3 and self.num_features == 1:
            x = x[:, :, 0]

        # Lấy embedding
        if self.num_features == 1:
            embeds = self.embedding(x)  # shape: (batch, seq_len, embed_dim)
        else:
            embeds = [self.embedding(x[:, :, i]) for i in range(self.num_features)]
            embeds = torch.cat(embeds, dim=2)  # shape: (batch, seq_len, embed_dim*num_features)

        # Xây dựng key padding mask:
        # Nếu x là tensor 3 chiều, giả sử x[:, :, 0] chứa indices; trường hợp khác tương tự.
        if x.dim() == 3:
            token_mask = (x[:, :, 0] != 0)
        else:
            token_mask = (x != 0)
        # True ở vị trí padding (xác định bởi ~token_mask)
        key_padding_mask = ~token_mask  # shape: (batch, seq_len)

        # Chuyển đổi embeds thành dạng (seq_len, batch, d_model)
        embeds = embeds.transpose(0, 1)  # (seq_len, batch, embed_dim)

        batch_size = embeds.size(1)
        # Thêm [CLS] token ở đầu chuỗi cho mỗi ví dụ
        cls_tokens = self.cls_token.expand(1, batch_size, self.d_model)
        embeds = torch.cat([cls_tokens, embeds], dim=0)  # (seq_len+1, batch, d_model)

        # Cập nhật key_padding_mask để bổ sung cho token [CLS] (luôn không bị masking)
        cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=key_padding_mask.device)
        key_padding_mask = torch.cat([cls_mask, key_padding_mask], dim=1)  # (batch, seq_len+1)

        # Áp dụng positional encoding
        embeds = self.pos_encoder(embeds)
        # TransformerEncoder nhận vào key_padding_mask với kích thước (batch, seq_len+1)
        transformer_out = self.transformer(embeds, src_key_padding_mask=key_padding_mask)
        
        # Lấy đầu ra của token [CLS]
        cls_output = transformer_out[0]  # shape: (batch, d_model)

        # Áp dụng normalization, dropout và fully connected layer cho phân lớp
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
        pe = pe.unsqueeze(1)  # shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x có kích thước: (seq_len, batch, d_model)
        seq_len = x.size(0)
        x = x + self.pe[:seq_len]
        return x
