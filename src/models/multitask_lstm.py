# src/models/multitask_lstm.py
import torch
import torch.nn as nn

class MultiTaskLSTM(nn.Module):
    """
    Multi-task LSTM: outputs (logit, reg) for classification and regression.
    """
    def __init__(self, input_dim, hidden=256, num_layers=3, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0.0)
        self.cls_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden,1))
        self.reg_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden,1))

    def forward(self, x):
        # x: [B, T, F]
        _, (h, _) = self.lstm(x)
        h = h[-1]  # [B, hidden]
        logit = self.cls_head(h).squeeze(-1)  # [B]
        reg = self.reg_head(h).squeeze(-1)    # [B]
        return logit, reg
