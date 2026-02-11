"""TransformerGamma: a lightweight Transformer model for self-collision safety prediction.

Given a joint-state vector [q, qd] of dimension 14, the model outputs a scalar
safety margin Gamma.  Larger Gamma indicates a safer configuration with respect
to self-collision.
"""

import torch
import torch.nn as nn


class TransformerGamma(nn.Module):
    """Transformer-based self-collision safety predictor.

    Parameters
    ----------
    input_dim : int
        Dimension of the input vector (default: 14 = 7 joint positions + 7 velocities).
    d_model : int
        Hidden dimension of the transformer encoder.
    nhead : int
        Number of attention heads.
    num_layers : int
        Number of transformer encoder layers.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        input_dim: int = 14,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.linear_encoder = nn.Linear(1, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(input_dim, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, q: torch.Tensor):
        """Forward pass.

        Parameters
        ----------
        q : torch.Tensor
            Input tensor of shape ``(B, 14)`` containing concatenated joint
            positions and velocities.

        Returns
        -------
        gamma_logits : torch.Tensor
            Raw logits of shape ``(B, 2)``.
        gamma : torch.Tensor
            Scalar safety margin ``gamma_logits[:, 0] - gamma_logits[:, 1]``,
            shape ``(B,)``.
        """
        q = q.unsqueeze(-1)                          # (B, 14, 1)
        x = self.linear_encoder(q)                   # (B, 14, d_model)
        x = x + self.positional_encoding.unsqueeze(0)
        x = self.transformer(x)
        x = x.mean(dim=1)                            # (B, d_model)
        gamma_logits = self.classifier(x)             # (B, 2)
        gamma = gamma_logits[:, 0] - gamma_logits[:, 1]
        return gamma_logits, gamma
