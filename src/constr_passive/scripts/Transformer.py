import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import os, rospkg, torch

class TransformerGamma(nn.Module):
    def __init__(self, input_dim=14, d_model=64, nhead=2, num_layers=4, dropout=0.1):
        super().__init__()
        self.linear_encoder = nn.Linear(1, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(input_dim, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, q):
        q = q.unsqueeze(-1)
        x = self.linear_encoder(q)
        x = x + self.positional_encoding.unsqueeze(0)
        x = self.transformer(x)
        x = x.mean(dim=1)
        gamma_logits = self.classifier(x)
        γ1, γ2 = gamma_logits[:, 0], gamma_logits[:, 1]
        Γ = γ1 - γ2
        return gamma_logits, Γ

device_sc   = torch.device("cpu")
_sc_model   = TransformerGamma().to(device_sc)
_sc_model2  = TransformerGamma().to(device_sc)

pkg_path = rospkg.RosPack().get_path('constr_passive')
ckpt     = os.path.join(pkg_path, 'models', 'network', 'transformer_gamma.pt')
ckpt2     = os.path.join(pkg_path, 'models', 'network', 'largesigularitydata_4e-2.pt')
_sc_model.load_state_dict(torch.load(ckpt, map_location=device_sc, weights_only=True))
_sc_model.eval()

_sc_model2.load_state_dict(torch.load(ckpt2, map_location=device_sc, weights_only=True))
_sc_model2.eval()

def gamma_model(
    q_batch: torch.Tensor,
    dq_batch: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """自碰撞 γ：数值越大越安全"""
    x = torch.cat([q_batch, dq_batch], dim=1).to(device_sc)
    x.requires_grad_(True)
    _, gamma = _sc_model(x)
    return gamma.squeeze(), x 

def gamma_model2(
    q_batch: torch.Tensor,
    dq_batch: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """自碰撞 γ：数值越大越安全"""
    x = torch.cat([q_batch, dq_batch], dim=1).to(device_sc)
    x.requires_grad_(True)
    _, gamma = _sc_model2(x)
    return gamma.squeeze(), x 

@torch.no_grad()
def gamma_only_model(q_batch: torch.Tensor, dq_batch: torch.Tensor) -> torch.Tensor:
    x = torch.cat([q_batch, dq_batch], dim=1).to(device_sc)
    _, gamma = _sc_model(x)
    return gamma.squeeze()