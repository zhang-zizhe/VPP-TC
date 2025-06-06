import torch
import torch.nn as nn

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
