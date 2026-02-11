"""Safety-related computations: acceleration bounds and self-collision gamma.

This module implements the viability-preserving acceleration bound algorithms
and the self-collision safety margin (Gamma) predictor used in VPP-TC.
"""

import math
import os
from typing import Optional, Tuple

import numpy as np
import torch

from vpptc.model import TransformerGamma

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_MODEL_PATH = os.path.join(
    _PACKAGE_DIR, os.pardir, "assets", "models", "transformer_gamma.pt"
)

# ---------------------------------------------------------------------------
# Module-level model singleton (loaded once on first call to gamma_model)
# ---------------------------------------------------------------------------

_sc_device = torch.device("cpu")
_sc_model: Optional[TransformerGamma] = None


def _load_gamma_model(model_path: str = None) -> TransformerGamma:
    """Load the pre-trained TransformerGamma model (lazy singleton)."""
    global _sc_model
    if _sc_model is not None:
        return _sc_model
    if model_path is None:
        model_path = _DEFAULT_MODEL_PATH
    _sc_model = TransformerGamma().to(_sc_device)
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
    """Evaluate the self-collision safety margin Gamma.

    Parameters
    ----------
    q_batch : torch.Tensor
        Joint positions, shape ``(B, 7)``.
    dq_batch : torch.Tensor
        Joint velocities, shape ``(B, 7)``.
    model_path : str, optional
        Path to the ``.pt`` weights file.  Uses the default pre-trained model
        if not specified.

    Returns
    -------
    gamma : torch.Tensor
        Safety margin (scalar for B=1), larger = safer.
    x : torch.Tensor
        Concatenated input ``[q, dq]`` with gradients enabled, useful for
        computing ``d(gamma)/d(q, dq)`` via back-propagation.
    """
    model = _load_gamma_model(model_path)
    x = torch.cat([q_batch, dq_batch], dim=1).to(_sc_device)
    x.requires_grad_(True)
    _, gamma = model(x)
    return gamma.squeeze(), x


def compute_gamma_and_grad(
    q: np.ndarray,
    qd: np.ndarray,
    threshold: float,
    model_path: str = None,
) -> Tuple[float, Optional[np.ndarray]]:
    """Compute Gamma and, if below *threshold*, its gradient w.r.t. [q, qd].

    Parameters
    ----------
    q : array-like, shape (7,)
        Joint positions.
    qd : array-like, shape (7,)
        Joint velocities.
    threshold : float
        If Gamma >= threshold the gradient is not computed (returns None).
    model_path : str, optional
        Override for the model weights path.

    Returns
    -------
    gamma_val : float
    grad_14 : np.ndarray or None
        Gradient of shape (14,) when gamma < threshold, else None.
    """
    q_t = torch.tensor(q, dtype=torch.float32, requires_grad=False)
    qd_t = torch.tensor(qd, dtype=torch.float32, requires_grad=False)

    gamma_t, x = gamma_model(q_t.unsqueeze(0), qd_t.unsqueeze(0), model_path)
    gamma_val = gamma_t.item()

    if gamma_val < threshold:
        gamma_t.backward()
        grad_14 = x.grad.squeeze(0).cpu().numpy()  # (14,)
        return gamma_val, grad_14
    return gamma_val, None


# ---------------------------------------------------------------------------
# Acceleration bound algorithms
# ---------------------------------------------------------------------------

def acc_bounds_from_pos(
    q: float, qd: float, qmin: float, qmax: float, dt: float
) -> Tuple[float, float]:
    """Position-limit-based acceleration bounds (Algorithm 1).

    Returns ``(qdd_lb, qdd_ub)`` such that the joint stays within
    ``[qmin, qmax]`` after one time step *dt*.
    """
    qddmax1 = -qd / dt
    qddmax2 = -(qd ** 2) / (2.0 * (qmax - q))
    qddmax3 = 2.0 * (qmax - q - dt * qd) / (dt ** 2)
    qddmin2 = (qd ** 2) / (2.0 * (q - qmin))
    qddmin3 = 2.0 * (qmin - q - dt * qd) / (dt ** 2)

    if qd >= 0:
        qdd_lb = qddmin3
        qdd_ub = qddmax3 if qddmax3 > qddmax1 else min(qddmax1, qddmax2)
    else:
        qdd_ub = qddmax3
        qdd_lb = qddmin3 if qddmin3 < qddmax1 else max(qddmax1, qddmin2)

    return qdd_lb, qdd_ub


def acc_bounds_from_viability(
    q: float, qd: float, qmin: float, qmax: float, qdd_max: float, dt: float
) -> Tuple[float, float]:
    """Viability-based acceleration bounds (Algorithm 2).

    Ensures the robot can always be brought to rest within ``[qmin, qmax]``
    given the maximum deceleration ``qdd_max``.
    """
    a = dt ** 2
    qdd1 = -qd / dt

    # Upper bound
    b = dt * (2.0 * qd + qdd_max * dt)
    c = qd ** 2 - 2.0 * qdd_max * (qmax - q - dt * qd)
    delta = b ** 2 - 4.0 * a * c
    if delta >= 0:
        root_ub = (-b + math.sqrt(delta)) / (2.0 * a)
        qdd_ub = max(qdd1, root_ub)
    else:
        qdd_ub = qdd1

    # Lower bound
    b = 2.0 * dt * qd - qdd_max * dt ** 2
    c = qd ** 2 - 2.0 * qdd_max * (q + dt * qd - qmin)
    delta = b ** 2 - 4.0 * a * c
    if delta >= 0:
        root_lb = (-b - math.sqrt(delta)) / (2.0 * a)
        qdd_lb = min(qdd1, root_lb)
    else:
        qdd_lb = qdd1

    return qdd_lb, qdd_ub


def compute_joint_acceleration_bounds(
    q: float,
    qd: float,
    qmin: float,
    qmax: float,
    qdot_max: float,
    qdd_max: float,
    dt: float,
    viability: bool = True,
) -> Tuple[float, float]:
    """Combined acceleration bounds for a single joint (Algorithm 3).

    Intersects position limits, velocity limits, viability limits, and the
    trivial acceleration limit ``|qdd| <= qdd_max``.
    """
    lb_pos, ub_pos = acc_bounds_from_pos(q, qd, qmin, qmax, dt)
    lb_vel = (-qdot_max - qd) / dt
    ub_vel = (qdot_max - qd) / dt

    if viability:
        lb_viab, ub_viab = acc_bounds_from_viability(q, qd, qmin, qmax, qdd_max, dt)
    else:
        lb_viab, ub_viab = -qdd_max, qdd_max

    lb_triv = -qdd_max
    ub_triv = qdd_max

    qdd_lb = max(lb_pos, lb_vel, lb_viab, lb_triv)
    qdd_ub = min(ub_pos, ub_vel, ub_viab, ub_triv)
    return qdd_lb, qdd_ub


def compute_joint_acceleration_bounds_vec(
    q: np.ndarray,
    qd: np.ndarray,
    qmin: np.ndarray,
    qmax: np.ndarray,
    qdot_max: np.ndarray,
    qdd_max: np.ndarray,
    dt: float,
    viability: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorised acceleration bounds over all joints.

    Parameters
    ----------
    q, qd : array-like, shape (n,)
        Current joint positions and velocities.
    qmin, qmax : array-like, shape (n,)
        Joint position limits.
    qdot_max : array-like, shape (n,)
        Joint velocity limits.
    qdd_max : array-like, shape (n,)
        Joint acceleration limits.
    dt : float
        Time step.
    viability : bool
        Whether to include viability constraints.

    Returns
    -------
    qdd_lb, qdd_ub : np.ndarray, shape (n,)
    """
    q = np.asarray(q)
    qd = np.asarray(qd)
    qmin = np.asarray(qmin)
    qmax = np.asarray(qmax)
    qdot_max = np.asarray(qdot_max)
    qdd_max = np.asarray(qdd_max)

    n = q.size
    qdd_lb = np.empty(n)
    qdd_ub = np.empty(n)

    for i in range(n):
        lb, ub = compute_joint_acceleration_bounds(
            q[i],
            qd[i],
            qmin[i],
            qmax[i],
            float(qdot_max[i]) if qdot_max.size > 1 else float(qdot_max),
            float(qdd_max[i]) if qdd_max.size > 1 else float(qdd_max),
            dt,
            viability,
        )
        qdd_lb[i] = lb
        qdd_ub[i] = ub

    return qdd_lb, qdd_ub
