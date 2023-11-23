import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_heads, dropout_attn, dropout_fc):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout_attn = dropout_attn
        self.dropout_fc = dropout_fc

        
        self.attention = nn.MultiheadAttention(self.embed_dim, self.num_heads, self.dropout_attn, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.fc = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.embed_dim),
            nn.GELU()
        )
        self.drop = nn.Dropout(self.dropout_fc)
    
    def forward(self, x):
        out, _ = self.attention(x, x, x)
        out = self.layer_norm1(out + x)
        x = self.fc(out)
        x = self.layer_norm2(x + out)
        return self.drop(x)

class Model(nn.Module):
    def __init__(self, input_size_vocab, out_size ,embed_dim, hidden_dim, num_heads, num_blocks, dropout_attn, dropout_fc):
        super().__init__()
        self.embeds = nn.Embedding(input_size_vocab, embed_dim)
        self.encoder_blocks = nn.ModuleList([Encoder(embed_dim, hidden_dim, num_heads, dropout_attn, dropout_fc) for _ in range(num_blocks)])
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout_fc),
            nn.Linear(2048, out_size),
            nn.LayerNorm(out_size)
        )
    def forward(self, x):
        x = self.embeds(x)
        for block in self.encoder_blocks:
            x = block(x)
        last_hidden = x
        out = self.decoder(last_hidden)
        return out
