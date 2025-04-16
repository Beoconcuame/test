import torch
import torch.nn as nn

class IpaBiGRUClassifierWithPos(nn.Module):

    def __init__(self, vocab_size, custom_pos_vocab_size, embedding_dim=300, hidden_dim=256,
                 num_layers=2, num_classes=2, dropout=0.5, combine_mode="sum"):
        super(IpaBiGRUClassifierWithPos, self).__init__()
        self.combine_mode = combine_mode.lower()
        if self.combine_mode not in ["sum", "concat"]:
            raise ValueError("combine_mode must be 'sum' or 'concat'")
            

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(custom_pos_vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        

        if self.combine_mode == "sum":
            gru_input_dim = embedding_dim
        else:  
            gru_input_dim = embedding_dim * 2
        

        self.gru = nn.GRU(
            input_size=gru_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim * 2, num_classes)
    
    def forward(self, tokens, custom_positions):


        token_embeds = self.embedding(tokens)

        pos_embeds = self.pos_embedding(custom_positions)
        

        if self.combine_mode == "concat":
            combined = torch.cat((token_embeds, pos_embeds), dim=-1)  
        else:
            combined = token_embeds + pos_embeds  
        
        combined = self.dropout(combined)
        

        out, hidden = self.gru(combined)
        
        h_forward = hidden[-2, :, :]  
        h_backward = hidden[-1, :, :]

        h = torch.cat((h_forward, h_backward), dim=1) 
        h = self.dropout(h)
        logits = self.fc(h)  
        return logits