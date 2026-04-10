"""Safety-related computations for the dual-arm OpenArm platform.

Self-collision safety margin (Gamma) predictor and viability-preserving
acceleration bounds.  No SDF / external collision avoidance.

The Gamma model takes 28-dim raw input (14 positions + 14 velocities).
No input normalisation — the model handles per-dimension scaling internally,
and raw gradients preserve the correct scaling for the QP avoidance constraint.
"""

import os
from typing import Optional, Tuple

import numpy as np
import torch

from vpptc.model import TransformerGamma
from vpptc.safety import (
    acc_bounds_from_pos,
    acc_bounds_from_viability,
    compute_joint_acceleration_bounds,
    compute_joint_acceleration_bounds_vec,
)

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODEL_PATH = os.path.join(
    _PACKAGE_DIR, os.pardir, "assets", "models", "transformer_gamma_dual_openarm.pt"
)

N_DOF = 14
INPUT_DIM = N_DOF * 2

# ---------------------------------------------------------------------------
# Module-level model singleton
# ---------------------------------------------------------------------------

_sc_device = torch.device("cpu")
_sc_model: Optional[TransformerGamma] = None


def _load_gamma_model(model_path: str = None) -> TransformerGamma:
    """Load the pre-trained dual-arm TransformerGamma model (lazy singleton)."""
    global _sc_model
    if _sc_model is not None:
        return _sc_model
    if model_path is None:
        model_path = _DEFAULT_MODEL_PATH
    _sc_model = TransformerGamma(input_dim=INPUT_DIM).to(_sc_device)
    _sc_model.load_state_dict(
        torch.load(model_path, map_location=_sc_device, weights_only=True)
    )
    _sc_model.eval()
    return _sc_model


def gamma_model(
    q_batch: torch.Tensor,
    dq_batch: torch.Tensor,
    model_path: str = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the dual-arm self-collision safety margin Gamma.

    Raw input is fed directly to the model (no normalisation).
    Gradients are w.r.t. the raw [q, dq] input.
    """
    model = _load_gamma_model(model_path)
    x_raw = torch.cat([q_batch, dq_batch], dim=1).to(_sc_device)
    x_raw.requires_grad_(True)
    _, gamma = model(x_raw)
    return gamma.squeeze(), x_raw


def compute_gamma_and_grad(
    q: np.ndarray,
    qd: np.ndarray,
    threshold: float,
    model_path: str = None,
) -> Tuple[float, Optional[np.ndarray]]:
    """Compute Gamma and, if below *threshold*, its gradient w.r.t. [q, qd].

    Returns
    -------
    gamma_val : float
    grad_28 : np.ndarray or None
        Gradient of shape (28,) w.r.t. raw [q, qd] when gamma < threshold.
    """
    q_t = torch.tensor(q, dtype=torch.float32, requires_grad=False)
    qd_t = torch.tensor(qd, dtype=torch.float32, requires_grad=False)

    gamma_t, x_raw = gamma_model(q_t.unsqueeze(0), qd_t.unsqueeze(0), model_path)
    gamma_val = gamma_t.item()

    if gamma_val < threshold:
        gamma_t.backward()
        grad_28 = x_raw.grad.squeeze(0).cpu().numpy()
        return gamma_val, grad_28
    return gamma_val, None
