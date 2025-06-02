#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Randomly sample q within given joint bounds, append fixed/random qd,
evaluate Gamma with TransformerGamma, and report collision stats.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

# -------------------------------------------------------------
# 1) 配置
# -------------------------------------------------------------
N_SAMPLES   = 50000          # 随机采样数量
THRESHOLD   = 2         # Γ 阈值 (< threshold 判碰撞)
OUTPUT_CSV  = "random_samples_checked.csv"

# 你的区间（按 q1_min,q1_max,q2_min,...）
bounds_line = (
    1.3720715, 1.3777591,
    0.92261535, 0.9263663,
   -0.3993555, -0.3958555,
   -2.777071, -2.774817,
    2.4965296, 2.5017796,
    3.152854, 3.1598537,
   -0.8598106, -0.8528106
)
bounds = np.array(bounds_line, dtype=np.float32).reshape(7, 2)  # shape (7,2)


fixed_qd = np.array([0.17917132, -0.1500672, 0.026805641,
                    -0.200802, 0.04455528, 0.16819422, -0.02755806],
                   dtype=np.float32)

class TransformerGamma(nn.Module):
    def __init__(self, input_dim=14, d_model=64, nhead=2,
                 num_layers=4, dropout=0.1):
        super().__init__()
        self.linear_encoder = nn.Linear(1, d_model)
        self.positional_encoding = nn.Parameter(
            torch.randn(input_dim, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=128,
            dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer,
                                                 num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = x.unsqueeze(-1)                  # (B,14,1)
        x = self.linear_encoder(x)
        x = x + self.positional_encoding.unsqueeze(0)
        x = self.transformer(x)
        x = x.mean(dim=1)                    # pool
        logits = self.classifier(x)
        gamma = logits[:, 0] - logits[:, 1]
        return logits, gamma

device = torch.device("cpu")
model = TransformerGamma().to(device)
model.load_state_dict(torch.load("transformer_gamma.pt",
                                 map_location=device))
model.eval()

@torch.no_grad()
def gamma_model(q_batch: torch.Tensor,
                dq_batch: torch.Tensor) -> torch.Tensor:
    x = torch.cat([q_batch, dq_batch], dim=1).to(device)
    _, gamma = model(x)
    return gamma.squeeze()


q_samples = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) \
            * np.random.rand(N_SAMPLES, 7).astype(np.float32)

qd_samples = np.repeat(fixed_qd[None, :], N_SAMPLES, axis=0)

q_tensor  = torch.tensor(q_samples, dtype=torch.float32)
qd_tensor = torch.tensor(qd_samples, dtype=torch.float32)

gamma_vals = gamma_model(q_tensor, qd_tensor).cpu().numpy()
is_collision = gamma_vals < THRESHOLD


n_coll = int(is_collision.sum())
print(f"Total samples : {N_SAMPLES}")
print(f"Collisions     : {n_coll}")
print(f"Safe rate      : {(1 - n_coll / N_SAMPLES) * 100:.2f}%")

# 保存到 CSV
df_out = pd.DataFrame(
    np.hstack([q_samples, qd_samples]),
    columns=[f"q{i+1}" for i in range(7)] +
            [f"qd{i+1}" for i in range(7)]
)
df_out["Gamma"] = gamma_vals
df_out["collision"] = is_collision.astype(int)
df_out.to_csv(OUTPUT_CSV, index=False)
print(f"Saved all samples to {OUTPUT_CSV}")
