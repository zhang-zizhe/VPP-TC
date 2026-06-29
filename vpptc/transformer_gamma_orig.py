"""TransformerGamma -- ORIGINAL VPP-TC self-collision predictor (faithful copy).

Extracted verbatim from the upstream repo so the new clean-viability 8M can be
trained with the ORIGINAL VPP-TC architecture+recipe as a baseline:
    https://github.com/zhang-zizhe/VPP-TC  ->  vpptc/model.py

This is intentionally a SEPARATE module from `vpptc/model.py` (which holds the
current single-head GammaRegressor).  Nothing here is shared with or grafted
onto GammaRegressor -- it is the original classifier, kept standalone.

What the original does
----------------------
A 2-class collision classifier.  Its *logit margin* is reused as the safety
margin Gamma that the VPP-TC controller treats as a barrier:

    gamma = logits[:, 0] - logits[:, 1]          # larger => safer

The controller fires avoidance when ``gamma < threshold`` and uses
``d(gamma)/d[q, qd]`` as the linear barrier direction.  So unlike a distance
regressor, here "Gamma" is in *logit units*, not metres.

Only change vs upstream: ``input_dim`` defaults to 28 (dual-arm OpenArm:
14 q + 14 qd) instead of 14 (single-arm Panda).  Architecture is identical
(Linear(1->d) value embedding, learned positional encoding, mean pooling,
MLP classifier head, dim_feedforward=128).

Sign convention for training labels (so gamma large => safe, matching how
simulate uses it): class 0 = SAFE, class 1 = UNSAFE, i.e.
``y = (min_dist < 0).long()``.
"""

import torch
import torch.nn as nn


class TransformerGamma(nn.Module):
    """Transformer-based self-collision safety predictor (original VPP-TC).

    Parameters
    ----------
    input_dim : int
        Input vector dimension (default 28 = 14 joint positions + 14 velocities
        for the dual-arm setup; upstream Panda used 14).
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
        input_dim: int = 28,
        d_model: int = 64,
        nhead: int = 2,
        num_layers: int = 4,
        dropout: float = 0.1,
        dim_feedforward: int = 128,
    ):
        super().__init__()
        self.linear_encoder = nn.Linear(1, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(input_dim, d_model))

        # Upstream hardcoded dim_feedforward=128 (a 2x expansion at d_model=64).
        # Exposed here so a wider d_model can scale the FFN too -- otherwise FFN
        # becomes a 1x (no-expansion) bottleneck and the extra width is wasted.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=dim_feedforward,
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
            Input tensor of shape ``(B, input_dim)`` containing concatenated
            joint positions and velocities.

        Returns
        -------
        gamma_logits : torch.Tensor
            Raw logits of shape ``(B, 2)``.
        gamma : torch.Tensor
            Safety margin ``gamma_logits[:, 0] - gamma_logits[:, 1]``, shape
            ``(B,)``.  Larger = safer.
        """
        q = q.unsqueeze(-1)                          # (B, input_dim, 1)
        x = self.linear_encoder(q)                   # (B, input_dim, d_model)
        x = x + self.positional_encoding.unsqueeze(0)
        x = self.transformer(x)
        x = x.mean(dim=1)                            # (B, d_model)
        gamma_logits = self.classifier(x)            # (B, 2)
        gamma = gamma_logits[:, 0] - gamma_logits[:, 1]
        return gamma_logits, gamma
