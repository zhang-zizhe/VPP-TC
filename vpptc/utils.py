"""Miscellaneous utility functions for VPP-TC."""

from typing import List

import numpy as np
import pybullet as p

# Panda joint acceleration limits: (lower, upper) for each of the 7 joints
JOINT_ACCELERATION_LIMITS = [
    (-15.0, 15.0),
    (-7.5, 7.5),
    (-10.0, 10.0),
    (-12.5, 12.5),
    (-15.0, 15.0),
    (-20.0, 20.0),
    (-20.0, 20.0),
]


def compute_qe(q, qd, acc_limits=None) -> List[float]:
    """Compute the joint positions when the robot decelerates to a full stop.

    For each joint, assuming maximum deceleration ``|a| = a_max``, the
    stopping position is:

        qe = q + 0.5 * qd * (|qd| / a_max)

    Parameters
    ----------
    q : array-like, shape (N,)
        Current joint positions.
    qd : array-like, shape (N,)
        Current joint velocities.
    acc_limits : list of (float, float) or None
        Per-joint acceleration limits ``(lower, upper)``.  When *None*,
        falls back to the single-arm ``JOINT_ACCELERATION_LIMITS``.

    Returns
    -------
    qe : list of float
        Predicted stopping positions for each joint.
    """
    if acc_limits is None:
        acc_limits = JOINT_ACCELERATION_LIMITS
    qe = []
    for j, vel in enumerate(qd):
        a_max = acc_limits[j][1]
        if vel == 0:
            qe.append(q[j])
        else:
            t_stop = abs(vel) / a_max
            delta = 0.5 * vel * t_stop
            qe.append(q[j] + delta)
    return qe


def compute_min_center_distance(
    robot_id: int,
    obstacle_id: int,
    sphere_radius: float,
    distance_threshold: float = 1.0,
) -> float:
    """Compute the minimum distance from any robot link surface to an obstacle centre.

    PyBullet ``getClosestPoints`` returns surface-to-surface distances.  For a
    sphere obstacle of radius *r*, the surface-to-centre distance is obtained
    by adding *r* back.

    Parameters
    ----------
    robot_id : int
        PyBullet body id of the robot.
    obstacle_id : int
        PyBullet body id of the spherical obstacle.
    sphere_radius : float
        Radius of the obstacle sphere.
    distance_threshold : float
        Search distance threshold for ``getClosestPoints``.

    Returns
    -------
    float
        Minimum link-surface to sphere-centre distance.
    """
    pts = p.getClosestPoints(
        bodyA=robot_id, bodyB=obstacle_id, distance=distance_threshold
    )
    if not pts:
        return float("inf")
    min_surface_dist = min(pt[8] for pt in pts)
    return min_surface_dist + sphere_radius


def feasible_qdd_region(
    grad_gamma: np.ndarray,
    qd: np.ndarray,
    dt: float,
    qdd_lb: np.ndarray,
    qdd_ub: np.ndarray,
) -> np.ndarray:
    """Find the acceleration that maximises Delta-Gamma within box constraints.

    The linear approximation is:

        Delta_gamma ~ g_eff^T @ qdd + c

    where ``g_eff = 0.5 * grad_q * dt^2 + grad_qd * dt`` and
    ``c = grad_q . qd * dt``.

    Parameters
    ----------
    grad_gamma : np.ndarray, shape (14,)
        Gradient of Gamma w.r.t. [q, qd].
    qd : np.ndarray, shape (7,)
        Current joint velocities.
    dt : float
        Time step.
    qdd_lb, qdd_ub : np.ndarray, shape (7,)
        Acceleration box constraints.

    Returns
    -------
    np.ndarray, shape (7,)
        Corner of the box that maximises Delta-Gamma.
    """
    grad_q = grad_gamma[:7]
    grad_qd = grad_gamma[7:]

    g_eff = 0.5 * grad_q * dt ** 2 + grad_qd * dt
    c = grad_q.dot(qd) * dt

    corner_min = np.where(g_eff > 0, qdd_lb, qdd_ub)
    corner_max = np.where(g_eff > 0, qdd_ub, qdd_lb)

    delta_gamma_min = np.dot(g_eff, corner_min) + c
    delta_gamma_max = np.dot(g_eff, corner_max) + c

    exists_positive = (delta_gamma_max > 0) and (delta_gamma_min < delta_gamma_max)
    if not exists_positive:
        return corner_max

    return corner_max
