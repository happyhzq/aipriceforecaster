import torch
import torch.nn as nn
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, num_layers: int = 2, dropout: float = 0.1, out_dim: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=num_layers, batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.head = nn.Sequential(
        nn.LayerNorm(hidden),
        nn.Linear(hidden, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        # x: [B, T, F]
        _, (h, _) = self.lstm(x)
        h = h[-1]  # [B, H]
        return self.head(h)  # [B, out_dim]
