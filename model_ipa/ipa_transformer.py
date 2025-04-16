import math
import torch
import torch.nn as nn

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):

        super().__init__()
        pe = torch.zeros(max_len, d_model)  
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)

    def forward(self, x):

        seq_len = x.size(1)
        return self.pe[:, :seq_len, :]

class TransformerWithCombinedPositionalEncoding(nn.Module):
    def __init__(self, vocab_size, custom_pos_vocab_size, d_model, nhead, num_layers,
                 dropout=0.1, max_len=5000, combination_mode='concat'):

        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.custom_pos_embedding = nn.Embedding(custom_pos_vocab_size, d_model)
        self.sinusoidal_encoding = SinusoidalPositionalEncoding(d_model, max_len=max_len)
        self.combination_mode = combination_mode.lower()

        if self.combination_mode == 'concat':
            self.proj = nn.Linear(d_model * 2, d_model)
        
        self.dropout = nn.Dropout(dropout)
        

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids, custom_positions, src_mask=None, src_key_padding_mask=None):

        token_embeds = self.token_embedding(token_ids)               
        standard_pos = self.sinusoidal_encoding(token_embeds)            
        custom_pos_embeds = self.custom_pos_embedding(custom_positions)    
        
        if self.combination_mode == 'sum':
            combined = token_embeds + standard_pos + custom_pos_embeds
        elif self.combination_mode == 'concat':
            base = token_embeds + standard_pos
            combined = torch.cat([base, custom_pos_embeds], dim=-1)      
            combined = self.proj(combined)                               
        else:
            raise ValueError("Unsupported combination mode. Choose 'sum' or 'concat'.")
        
        combined = self.dropout(combined)
    
        combined = combined.transpose(0, 1)  
        transformer_output = self.transformer_encoder(combined, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        transformer_output = transformer_output.transpose(0, 1)  
        
        logits = self.fc_out(transformer_output)  
        return logits
