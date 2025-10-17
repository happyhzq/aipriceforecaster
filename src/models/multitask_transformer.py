import torch.nn as nn
class MultiTaskTransformer(nn.Module):
    def __init__(self, input_dim, d_model=256, nhead=8, num_layers=3, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.cls = nn.Linear(d_model,1)
        self.reg = nn.Linear(d_model,1)
    def forward(self, x):
        z = self.inp(x)
        z = self.encoder(z)
        h = z.mean(1)
        logit = self.cls(h).squeeze(-1)
        reg = self.reg(h).squeeze(-1)
        return logit, reg