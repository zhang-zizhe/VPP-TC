import torch
from torch import nn

class extransformer_reg(nn.Module):
    def __init__(self, d_model=64, nhead=2, layers=4, dropout=0.1):
        super().__init__()
        self.enc  = nn.Linear(1, d_model)
        self.pe   = nn.Parameter(torch.randn(14, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True)
        self.tf   = nn.TransformerEncoder(enc, layers)
        self.reg  = nn.Sequential(nn.Linear(d_model,64), nn.ReLU(),
                                  nn.Linear(64,1))
    def forward(self, q):                 # (B,14)
        x = self.enc(q.unsqueeze(-1)) + self.pe
        x = self.tf(x).mean(dim=1)
        return self.reg(x).squeeze(1)