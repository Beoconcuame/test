import torch
import torch.nn as nn
import torch.nn.functional as F

class IpaTextCNNClassifierWithPos(nn.Module):

    def __init__(self, vocab_size, custom_pos_vocab_size, embedding_dim=300, num_classes=2, 
                 filter_sizes=[3, 4, 5], num_filters=100, dropout=0.5, combine_mode="sum"):
        super(IpaTextCNNClassifierWithPos, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.pos_embedding = nn.Embedding(custom_pos_vocab_size, embedding_dim)
        self.combine_mode = combine_mode
        

        if combine_mode == "concat":
            conv_input_dim = embedding_dim * 2
        else:  
            conv_input_dim = embedding_dim
        

        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (fs, conv_input_dim))
            for fs in filter_sizes
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)
    
    def forward(self, tokens, custom_positions):

        token_embeds = self.embedding(tokens)            
        pos_embeds = self.pos_embedding(custom_positions)  
        

        if self.combine_mode == "concat":
            x = torch.cat((token_embeds, pos_embeds), dim=-1)  
        else:
            x = token_embeds + pos_embeds                      
        
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