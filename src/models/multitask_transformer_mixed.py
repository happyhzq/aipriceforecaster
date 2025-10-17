# src/models/multitask_transformer_mixed.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ChannelSE(nn.Module):
    """Squeeze-and-Excitation style channel attention for time-series features (per position pooling)."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
    def forward(self, x):
        # x: [B, T, C]
        s = x.mean(dim=1)  # [B, C]
        s = torch.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s.unsqueeze(1)  # broadcast

class LocalSelfAttention(nn.Module):
    """
    Local-window self-attention: compute attention only within +/- window radius.
    Simpler implementation: compute full attention then mask for distance > window.
    """
    def __init__(self, d_model, nhead, local_window: int = 8, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
        assert d_model % nhead == 0
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)
        self.local_window = local_window
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, T, C]
        B, T, C = x.shape
        qkv = self.qkv(x)  # [B, T, 3C]
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # [B, nhead, T, head_dim]
        q = q.view(B, T, self.nhead, self.head_dim).permute(0,2,1,3)
        k = k.view(B, T, self.nhead, self.head_dim).permute(0,2,1,3)
        v = v.view(B, T, self.nhead, self.head_dim).permute(0,2,1,3)
        # compute scores
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.head_dim)  # [B, nhead, T, T]
        # mask out distances > local_window
        if self.local_window is not None and self.local_window < T:
            idx = torch.arange(T, device=x.device)
            dist = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs()  # [T, T]
            mask = (dist > self.local_window).to(x.device)  # True -> masked
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B, nhead, T, head_dim]
        out = out.permute(0,2,1,3).contiguous().view(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class MixedAttentionBlock(nn.Module):
    """
    One block: (1) local self-attention, (2) global multihead self-attention, (3) channel SE
    Each followed by residual + layernorm.
    """
    def __init__(self, d_model=256, nhead=8, local_window=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.local_attn = LocalSelfAttention(d_model, nhead, local_window, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.global_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Linear(dim_feedforward, d_model))
        self.norm3 = nn.LayerNorm(d_model)
        self.se = ChannelSE(d_model, reduction=8)

    def forward(self, x):
        # x: [B, T, C]
        # local
        y = self.local_attn(x)
        x = self.norm1(x + y)
        # global (full) self-attention
        attn_out, _ = self.global_attn(x, x, x, need_weights=False)
        x = self.norm2(x + attn_out)
        # channel SE
        x = x + self.se(x)
        # feedforward
        x = self.norm3(x + self.ff(x))
        return x

class MultiTaskTransformerMixed(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=4, local_window=8, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.blocks = nn.ModuleList([MixedAttentionBlock(d_model, nhead, local_window, dim_feedforward, dropout) for _ in range(num_layers)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.cls_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))
        self.reg_head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, x):
        # x: [B, T, F]
        z = self.input_proj(x)  # [B, T, d_model]
        for b in self.blocks:
            z = b(z)
        # pool in time
        zt = z.transpose(1,2)  # [B, C, T]
        h = self.pool(zt).squeeze(-1)  # [B, C]
        logit = self.cls_head(h).squeeze(-1)
        reg = self.reg_head(h).squeeze(-1)
        return logit, reg
