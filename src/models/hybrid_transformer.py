# src/models/hybrid_transformer.py
import torch, torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, d_model: int, r: int = 8):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model//r),
            nn.ReLU(inplace=True),
            nn.Linear(d_model//r, d_model),
            nn.Sigmoid()
        )
    def forward(self, x):  # (B,T,C)
        B,T,C = x.shape
        s = x.mean(dim=1)            # (B,C)
        w = self.fc(s).unsqueeze(1)  # (B,1,C)
        return x * w

class LocalConv(nn.Module):
    def __init__(self, d_model: int, k: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=k, padding=k//2, groups=d_model)
    def forward(self, x):  # (B,T,C)
        z = x.transpose(1,2)         # (B,C,T)
        z = self.conv(z)
        return z.transpose(1,2)

class HybridBlock(nn.Module):
    def __init__(self, d_model=384, nhead=8, dim_ff=768, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.local = LocalConv(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.ca = ChannelAttention(d_model)

    def forward(self, x):  # (B,T,C)
        z = self.local(self.norm1(x)) + x
        a, _ = self.attn(z, z, z, need_weights=False)
        z = a + z
        z = self.ff(self.norm2(z)) + z
        z = self.ca(z)
        return z

class HybridTransformerMTL(nn.Module):
    def __init__(self, input_dim, d_model=384, nhead=8, num_layers=6, dim_ff=768, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([HybridBlock(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.cls = nn.Linear(d_model, 1)
        self.reg = nn.Linear(d_model, 1)

    def forward(self, x):  # (B,T,F)
        z = self.inp(x)
        for blk in self.layers:
            z = blk(z)
        h = self.norm(z).mean(dim=1)
        logit = self.cls(h).squeeze(-1)
        reg = self.reg(h).squeeze(-1)
        return logit, reg
