import torch
import torch.nn as nn

class TransformerExtractor(nn.Module):
    def __init__(self,
                 input_dim: int = 2,
                 d_model: int = 128, #64
                 dropout: float = 0.1,
                 nhead: int = 8, #4
                 dim_feedforward: int = 256, #128
                 num_layers: int = 3, #2
                 seq_len: int = 40):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        # positional encoding
        self.pos_embedding = nn.parameter.Parameter(torch.randn(1, seq_len, d_model))
        # Transformer Encoder
        self.encoder_layer = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)])
        self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d_model))


    def get_position_encoding(self, seq_len, d_model):
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, relative_vecs): # (1,40,2)
        #print("relative_vecs", relative_vecs.min().item(), relative_vecs.max().item(), relative_vecs.mean().item(), relative_vecs.std().item())
        mask = (relative_vecs.abs().sum(dim=-1) == 0)
        x = self.input_proj(relative_vecs)
        x = self.input_norm(x)
        x = x + self.pos_embedding

        for layer in self.encoder_layer:
            x = layer(x, src_key_padding_mask=mask)
        batch_size = x.size(0)
        query = self.pool_query.expand(batch_size, -1, -1)
        pooled,_ = self.attention_pool(query, x, x, key_padding_mask=mask)

        return pooled.squeeze(1)