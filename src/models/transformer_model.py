import torch
import torch.nn as nn

class TransformerForecaster(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 8, num_layers: int = 3, dim_feedforward: int = 256, dropout: float = 0.1, out_dim: int = 1):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim)
        )

    def forward(self, x):
        # x: [B, T, F]
        z = self.in_proj(x)                # [B, T, d]
        z = self.encoder(z)                # [B, T, d]
        z = z.transpose(1,2)               # [B, d, T]
        z = self.pool(z).squeeze(-1)       # [B, d]
        return self.head(z)                # [B, out_dim]
