import math
import numpy as np
from typing import Tuple, List, Union

def acc_bounds_from_pos(q, qd, qmin, qmax, dt):
    qddmax1 = -qd/dt
    qddmax2 = -qd**2/(2*(qmax-q))
    qddmax3 = 2*(qmax-q-dt*qd)/(dt**2)
    qddmin2 = qd**2/(2*(q-qmin))
    qddmin3 = 2 * (qmin - q - dt * qd) / (dt**2)

    if qd >= 0:
        qdd_lb = qddmin3
        if qddmax3 > qddmax1:
            qdd_ub = qddmax3
        else:
            qdd_ub = min(qddmax1, qddmax2)
    else:
        qdd_ub = qddmax3
        if qddmin3 < qddmax1:
            qdd_lb = qddmin3
        else:
            qdd_lb = max(qddmax1, qddmin2)

    return qdd_lb, qdd_ub

def acc_bounds_from_viability(q, qd, qmin, qmax, qdd_max, dt):
    """
    Compute acceleration bounds (qdd_lb, qdd_ub) for state viability 
    w.r.t. position [qmin, qmax] using Algorithm 2 from the paper.
    """
    # Precompute common terms
    a = dt**2
    # Upper‐bound quadratic coeffs
    b = dt * (2*qd + qdd_max * dt)
    c = qd**2 - 2 * qdd_max * (qmax - q - dt*qd)
    qdd1 = -qd / dt

    # Discriminant and UB root
    delta = b**2 - 4 * a * c
    if delta >= 0:
        root_ub = (-b + math.sqrt(delta)) / (2 * a)
        qdd_ub = max(qdd1, root_ub)
    else:
        qdd_ub = qdd1

    # Lower‐bound quadratic coeffs
    b = 2 * dt * qd - qdd_max * dt**2
    c = qd**2 - 2 * qdd_max * (q + dt*qd - qmin)
    delta = b**2 - 4 * a * c
    if delta >= 0:
        root_lb = (-b - math.sqrt(delta)) / (2 * a)
        qdd_lb = min(qdd1, root_lb)
    else:
        qdd_lb = qdd1

    return qdd_lb, qdd_ub

def compute_joint_acceleration_bounds(q, qd, qmin, qmax, qdot_max, qdd_max, dt, viability=True):
    """
    Algorithm 3: Compute joint acceleration bounds by combining
    1) position limits via acc_bounds_from_pos,
    2) velocity limits,
    3) viability limits via acc_bounds_from_viability,
    4) trivial |qdd| ≤ qdd_max constraint.
    Returns (qdd_lb, qdd_ub).
    """
    # 1) Position-based bounds
    lb_pos, ub_pos = acc_bounds_from_pos(q, qd, qmin, qmax, dt)

    # 2) Velocity-based bounds (assume symmetric ±qdot_max)
    lb_vel = (-qdot_max - qd) / dt
    ub_vel = ( qdot_max - qd) / dt

    # 3) Viability-based bounds
    if viability:
        lb_viab, ub_viab = acc_bounds_from_viability(q, qd, qmin, qmax, qdd_max, dt)
    else:
        lb_viab = -qdd_max
        ub_viab =  qdd_max
    # If viability is not used, we can set the bounds to trivial limits

    # 4) Trivial acceleration limits
    lb_triv = -qdd_max
    ub_triv =  qdd_max

    # Combine all lower‐bounds (take max) and upper‐bounds (take min)
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
    viability: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorized over n joints. Returns (qdd_lb, qdd_ub), each an array of length n.
    Each element i is compute_joint_acceleration_bounds(q[i], qd[i], ..., dt).
    """
    # ensure numpy arrays
    q         = np.asarray(q)
    qd        = np.asarray(qd)
    qmin      = np.asarray(qmin)
    qmax      = np.asarray(qmax)
    qdot_max  = np.asarray(qdot_max)
    qdd_max   = np.asarray(qdd_max)
    viability = bool(viability)

    n = q.size
    qdd_lb = np.empty(n)
    qdd_ub = np.empty(n)

    for i in range(n):
        lb, ub = compute_joint_acceleration_bounds(
            q[i], qd[i],
            qmin[i], qmax[i],
            qdot_max[i] if qdot_max.size>1 else float(qdot_max),
            qdd_max[i]  if qdd_max.size>1  else float(qdd_max),
            dt,
            viability
        )
        qdd_lb[i] = lb
        qdd_ub[i] = ub

    return qdd_lb, qdd_ub

# def compute_torque_bounds(robot, qdd_lb: np.ndarray, qdd_ub: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
#     """
#     Given:
#       - robot.q and robot.qd already set on an RTB DHRobot/ERobot
#       - qdd_lb, qdd_ub: (n,) arrays of min/max joint accelerations
#     Compute the torque interval [tau_min, tau_max] per joint via:
#         tau = M(q)*qdd + C(q,qd) + G(q)
#     and taking elementwise min/max over the two acceleration extremes.

#     Returns:
#       tau_min, tau_max : two (n,) arrays
#     """
#     # 1. Dynamics terms at the current state
#     M = robot.inertia(robot.q)             # (n×n)
#     print("shape of M", M.shape)
#     C = robot.coriolis(robot.q, robot.qd)  # (n,)
#     print("shape of C", C.shape)
#     G = robot.gravload(robot.q)            # (n,)
#     print("shape of G", G.shape)
#     # 2. Torque at lower‐ and upper‐accel bounds
#     tau_lb = M @ qdd_lb + C @ robot.qd + G
#     print("tau_lb", tau_lb) 
#     tau_ub = M @ qdd_ub + C @ robot.qd + G
#     print("tau_ub", tau_ub)

#     # 3. Final safe torque interval
#     tau_min = np.minimum(tau_lb, tau_ub)
    
#     tau_max = np.maximum(tau_lb, tau_ub)

#     return tau_min, tau_max


# def random_pos_near_limits(qmin, qmax, margin=0.05):
#     """
#     For each joint i:
#       – with 50% chance pick near the lower limit: uniform in [qmin[i], qmin[i] + m*(qmax[i]-qmin[i])]
#       – otherwise pick near the upper limit: uniform in [qmax[i] - m*(qmax[i]-qmin[i]), qmax[i]]
#     """
#     qmin = np.asarray(qmin)
#     qmax = np.asarray(qmax)
#     n = qmin.size
#     span = qmax - qmin
#     low_vals  = qmin + np.random.rand(n) * (margin * span)
#     high_vals = qmax - np.random.rand(n) * (margin * span)
#     # choose for each joint whether to use low or high region
#     mask = np.random.rand(n) < 0.5
#     return np.where(mask, low_vals, high_vals)

# def random_vel_near_limits(qdlim, margin=0.05):
#     """
#     For each joint i (with symmetric [-qdlim, +qdlim]):
#       – 50% chance near -limit: uniform in [-qdlim[i], -qdlim[i] + 2*m*qdlim[i]]
#       – otherwise near +limit: uniform in [qdlim[i] - 2*m*qdlim[i], +qdlim[i]]
#     """
#     qdlim = np.asarray(qdlim)
#     n = qdlim.size
#     # total vel span is 2*qdlim
#     span = 2 * qdlim
#     low_vals  = -qdlim + np.random.rand(n) * (margin * span)
#     high_vals =  qdlim - np.random.rand(n) * (margin * span)
#     mask = np.random.rand(n) < 0.5
#     return np.where(mask, low_vals, high_vals)