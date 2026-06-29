"""GammaRegressor: single-head distance regression model for OpenArm.

Design choice (vs the old TransformerGamma dual-head):
  * One head, one truth.  We predict the signed min self-collision
    distance `gamma` directly.  Classification is a post-hoc thresholding
    on `gamma`, so cls and reg can never disagree.
  * Output is LINEAR by default (gamma = raw, unbounded).  The earlier
    `tanh * max_dist` squash is itself a soft clip: it caps what the model
    can express AND zeroes the gradient d(gamma)/dq in saturation -- worst
    exactly in deep penetration, where the controller most needs a strong
    push-out gradient.  Linear output keeps the gradient constant
    everywhere and lets deep-penetration labels be learned exactly.  The
    Huber loss (linear, not squared, on large errors) already handles the
    heavy tail that tanh was originally meant to tame.  `output_mode="tanh"`
    remains available for loading legacy checkpoints.
  * CLS token + transformer pool replaces the old mean-pool (which
    averaged away positional encoding, making per-joint structure
    invisible to the head).
  * Per-joint token embedding + small positional encoding init
    (std=0.02) so the position signal does not drown the value signal
    at step 0.
  * Pre-norm transformer block + FFN 4 * d_model for stable training
    of a small transformer without aggressive lr scheduling.

Why no classifier head:
  The old dual-head model had:
    loss = CE(logits, label) + lambda * MSE(gamma, true_d)
  Same task ("is d<0?" and "what is d?") split into two heads competing
  for the same trunk.  The cls head wants a boundary-sensitive
  representation; the reg head wants a global-distance-aware one.
  These are incompatible -> trunk oscillates -> rec_unsafe never
  stabilizes.  With a single regression head, the loss surface is one
  consistent objective and the boundary classification is derived,
  not learned separately.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GammaRegressor(nn.Module):
    """Single-head signed-distance regressor.

    Parameters
    ----------
    input_dim : int
        Number of scalar inputs.  For OpenArm dual-arm with (q, qd):
        14 + 14 = 28.
    d_model : int
        Hidden width of the transformer.
    nhead : int
        Number of attention heads (must divide d_model).
    num_layers : int
        Transformer encoder depth.
    dropout : float
        Dropout in attention + FFN.
    max_dist : float
        Only used when output_mode == "tanh": output is squashed via tanh
        to (-max_dist, +max_dist).  NOTE: this squash is itself a (soft)
        clip -- it caps what the model can express and kills the gradient
        d(gamma)/dq in saturation (worst exactly where the controller most
        needs it, e.g. deep penetration).  Kept only for backward-compat
        with old checkpoints.
    output_mode : str
        "linear" (default): gamma = raw, fully unbounded -> NO clip on the
            output, constant gradient scale everywhere.  Use this for the
            no-clip viability data (deep penetration kept exact).
        "tanh": legacy gamma = tanh(raw) * max_dist.
    """

    def __init__(
        self,
        input_dim: int = 28,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dropout: float = 0.1,
        max_dist: float = 0.20,
        output_mode: str = "linear",
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        assert output_mode in ("linear", "tanh"), \
            f"output_mode must be 'linear' or 'tanh', got {output_mode!r}"
        self.input_dim = input_dim
        self.max_dist = max_dist
        self.output_mode = output_mode

        # Per-token value embedding: maps scalar q_i -> d_model.
        # Shared across tokens (positional info comes from joint_emb).
        self.token_emb = nn.Linear(1, d_model)

        # Learnable per-joint embedding (analogous to position embedding
        # in NLP).  Small init so it doesn't dominate value signal early.
        self.joint_emb = nn.Parameter(
            torch.randn(input_dim, d_model) * 0.02
        )

        # CLS token -- pooled to head, replaces mean(dim=1) which
        # destroyed positional structure.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,   # standard 4x expansion
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,               # pre-norm: more stable
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        """Predict signed min self-collision distance.

        Parameters
        ----------
        q : (B, input_dim) tensor
            Joint state features.

        Returns
        -------
        gamma : (B,) tensor
            Signed distance.  Positive = safe, negative = penetration.
            output_mode="linear": unbounded (no clip). "tanh": bounded to
            (-max_dist, +max_dist).
        """
        B = q.size(0)
        # (B, T) -> (B, T, 1) -> (B, T, d_model)
        x = self.token_emb(q.unsqueeze(-1)) + self.joint_emb
        # Prepend CLS token: (B, T+1, d_model)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.transformer(x)
        # Pool CLS slot
        raw = self.head(x[:, 0]).squeeze(-1)        # (B,)
        if self.output_mode == "tanh":
            return torch.tanh(raw) * self.max_dist
        # linear: fully unbounded -> NO clip, constant d(gamma)/d(raw)=1
        return raw

    @torch.no_grad()
    def predict_safe(self, q: torch.Tensor, margin: float = 0.0):
        """Convenience: threshold-based safety classification.

        Parameters
        ----------
        margin : float
            Required clearance.  margin=0 means "no penetration";
            margin=0.005 reserves a 5 mm safety buffer.

        Returns
        -------
        is_safe : (B,) bool tensor
        gamma   : (B,) tensor of predicted distances
        """
        gamma = self.forward(q)
        is_safe = gamma > margin
        return is_safe, gamma


# ---------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------

def asymmetric_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    delta: float = 0.005,
    collision_weight: float = 10.0,
    boundary_radius: float = 0.02,
    boundary_weight: float = 3.0,
    over_pred_weight: float = 1.0,
) -> torch.Tensor:
    """Huber loss with sample-wise weighting for the OpenArm distance task.

    Three reweighting axes that the old MSE+CE setup completely missed:

    1. Collision samples (target < 0) are rare but DECISION-critical;
       upweight by `collision_weight` (~= safe/unsafe ratio in dataset).
    2. Near-boundary samples (|target| < boundary_radius) are where the
       regressor must be most accurate to support classification.
       Upweight by `boundary_weight` on top of (1).
    3. DIRECTIONAL safety asymmetry: over-prediction (pred > target) means
       the model claims MORE clearance than truly exists -- i.e. it lies
       about safety, the failure mode that lets VPP-TC drive the arms into
       contact.  Under-prediction is merely a (safe) false alarm.  When
       `over_pred_weight > 1`, optimistic residuals are penalized harder so
       the model is pushed to err conservative even where it can't fit the
       sharp finger-vs-link distance field exactly.

    Huber (vs MSE) prevents far-field saturation samples (which are all
    pinned near +max_dist) from dominating gradient via their squared
    residuals.

    Returns a scalar loss; the denominator is sum-of-weights, so the
    expected value is invariant to dataset class balance.
    """
    diff = pred - target
    abs_diff = diff.abs()
    huber = torch.where(
        abs_diff < delta,
        0.5 * diff * diff / delta,
        abs_diff - 0.5 * delta,
    )

    w_coll = torch.where(
        target < 0.0,
        torch.full_like(target, collision_weight),
        torch.ones_like(target),
    )
    w_bnd = torch.where(
        target.abs() < boundary_radius,
        torch.full_like(target, boundary_weight),
        torch.ones_like(target),
    )
    # diff > 0  <=>  pred > target  <=>  model over-predicts clearance (unsafe)
    w_dir = torch.where(
        diff > 0.0,
        torch.full_like(target, over_pred_weight),
        torch.ones_like(target),
    )
    w = w_coll * w_bnd * w_dir
    return (huber * w).sum() / w.sum().clamp_min(1e-8)
