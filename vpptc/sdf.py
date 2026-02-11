"""Signed Distance Field (SDF) query wrappers for external collision avoidance.

This module provides a thin interface over the vendored RDF (Robot Distance
Fields) library, supporting both single-point and batched SDF queries with
analytical joint-space gradients.
"""

import os
from typing import Tuple

import numpy as np
import torch

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_RDF_MODEL = os.path.join(
    _PACKAGE_DIR, os.pardir, "assets", "models", "rdf", "BP_24.pt"
)

# ---------------------------------------------------------------------------
# Lazy-loaded globals
# ---------------------------------------------------------------------------

_device: str = None
_sdf_model = None
_panda_layer = None
_bpsdf = None


def _ensure_loaded(model_path: str = None) -> None:
    """Lazily initialise the RDF model, PandaLayer, and BPSDF engine."""
    global _device, _sdf_model, _panda_layer, _bpsdf

    if _bpsdf is not None:
        return

    if model_path is None:
        model_path = _DEFAULT_RDF_MODEL

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    # Import from the vendored third-party RDF package
    from third_party.rdf.bf_sdf import BPSDF
    from third_party.rdf.panda_layer.panda_layer import PandaLayer

    _sdf_model = torch.load(model_path, map_location=_device, weights_only=False)
    _panda_layer = PandaLayer(_device)
    _bpsdf = BPSDF(
        n_func=24,
        domain_min=-1.0,
        domain_max=1.0,
        robot=_panda_layer,
        device=_device,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def query_sdf(
    x_np: np.ndarray,
    pose_np: np.ndarray,
    theta_np: np.ndarray,
    model_path: str = None,
) -> Tuple[float, np.ndarray]:
    """Query the whole-body SDF for a single point and joint configuration.

    Parameters
    ----------
    x_np : np.ndarray, shape (1, 3) or (3,)
        Query point in world frame.
    pose_np : np.ndarray, shape (4, 4)
        Base pose of the robot.
    theta_np : np.ndarray, shape (7,)
        Joint configuration.
    model_path : str, optional
        Path to the pre-trained RDF model (``.pt``).

    Returns
    -------
    dst : float
        Signed distance value (positive = outside the robot).
    grad_q : np.ndarray, shape (7,)
        Gradient of the SDF w.r.t. joint angles.
    """
    _ensure_loaded(model_path)

    x_t = torch.from_numpy(x_np.reshape(1, 3).astype(np.float32)).to(_device)
    pose_t = torch.from_numpy(pose_np.reshape(1, 4, 4).astype(np.float32)).to(_device)
    theta_t = (
        torch.from_numpy(theta_np.reshape(1, 7).astype(np.float32))
        .to(_device)
        .requires_grad_(True)
    )

    with torch.no_grad():
        sdf, d_sdf = _bpsdf.get_whole_body_sdf_with_joints_grad_batch(
            x_t, pose_t, theta_t, _sdf_model
        )
        grad_q = d_sdf.squeeze(0).squeeze(0).cpu().numpy()  # (7,)
        dst = float(sdf.squeeze().item())

    return dst, grad_q


def query_sdf_batch(
    x_np: np.ndarray,
    pose_np: np.ndarray,
    theta_np: np.ndarray,
    model_path: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batched SDF query for multiple points and joint configurations.

    Parameters
    ----------
    x_np : np.ndarray, shape (N, 3)
        Query points in world frame.
    pose_np : np.ndarray, shape (B, 4, 4) or (4, 4)
        Base poses.
    theta_np : np.ndarray, shape (B, 7)
        Joint configurations.
    model_path : str, optional
        Path to the pre-trained RDF model.

    Returns
    -------
    dsts : np.ndarray, shape (B, N)
        SDF values for each (theta, point) pair.
    grad_qs : np.ndarray, shape (B, N, 7)
        Gradients of the SDF w.r.t. joint angles.
    """
    _ensure_loaded(model_path)

    x_np = np.asarray(x_np, dtype=np.float32).reshape(-1, 3)
    theta_np = np.asarray(theta_np, dtype=np.float32)
    if theta_np.ndim == 1:
        theta_np = theta_np.reshape(1, 7)
    B = theta_np.shape[0]

    pose_np = np.asarray(pose_np, dtype=np.float32)
    if pose_np.shape == (4, 4):
        pose_np = np.broadcast_to(pose_np, (B, 4, 4)).copy()

    x_t = torch.from_numpy(x_np).to(_device)
    pose_t = torch.from_numpy(pose_np).to(_device)
    theta_t = torch.from_numpy(theta_np).to(_device)

    with torch.no_grad():
        sdf, d_sdf = _bpsdf.get_whole_body_sdf_with_joints_grad_batch(
            x_t, pose_t, theta_t, _sdf_model
        )
        dsts = sdf.detach().cpu().numpy().astype(np.float32)       # (B, N)
        grad_qs = d_sdf.detach().cpu().numpy().astype(np.float32)  # (B, N, 7)

    return dsts, grad_qs
