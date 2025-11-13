import torch
import torch.nn as nn

class TransformerExtractor(nn.Module):
    def __init__(self,
                 input_dim: int = 2,
                 d_model: int = 64,
                 dropout: float = 0.1,
                 nhead: int = 4,
                 dim_feedforward: int = 128,
                 num_layers: int = 2,
                 seq_len: int = 40):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)

        # positional encoding
        pe = self.get_position_encoding(seq_len, d_model)
        self.register_buffer('pos_enc', pe)

        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers)
        self.dropout = nn.Dropout(p=dropout)

    def get_position_encoding(self, seq_len, d_model):
        # TODO:  reshape
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
        #print("input_proj", x.min().item(), x.max().item(), x.mean().item(), x.std().item())
        x = x + self.pos_enc.unsqueeze(0)
        x = x.permute(1, 0, 2)
        x = self.dropout(x)
        x = self.transformer(x, src_key_padding_mask=mask)
        x = x.permute(1, 0, 2)
        x = x.mean(dim=1)
        return x
