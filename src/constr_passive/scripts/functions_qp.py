import math
import numpy as np
from typing import Tuple, List, Union, Optional


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
    # print("pos lb, ub:", lb_pos, ub_pos)

    # 2) Velocity-based bounds (assume symmetric ±qdot_max)
    lb_vel = (-qdot_max - qd) / dt
    ub_vel = ( qdot_max - qd) / dt
    # print("vel lb, ub:", lb_vel, ub_vel)

    # 3) Viability-based bounds
    if viability:
        lb_viab, ub_viab = acc_bounds_from_viability(q, qd, qmin, qmax, qdd_max, dt)
        # print("viab lb, ub:", lb_viab, ub_viab)
    else:
        lb_viab = -qdd_max
        ub_viab =  qdd_max
    # If viability is not used, we can set the bounds to trivial limits

    # 4) Trivial acceleration limits
    lb_triv = -qdd_max
    ub_triv =  qdd_max
    # print("triv lb, ub:", lb_triv, ub_triv)

    # Combine all lower‐bounds (take max) and upper‐bounds (take min)
    qdd_lb = max(lb_pos, lb_vel, lb_viab, lb_triv)
    qdd_ub = min(ub_pos, ub_vel, ub_viab, ub_triv)
    if qdd_lb == lb_pos:
        flag = 1
        print(lb_pos, ub_pos)
    elif qdd_lb == lb_vel:
        flag = 2
        print(lb_vel, ub_vel)
    elif qdd_lb == lb_viab:
        flag = 3
        print(lb_viab, ub_viab)
    else:
        flag = 0

    return qdd_lb, qdd_ub, flag

def compute_joint_acceleration_bounds_vec(
    q: np.ndarray,
    qd: np.ndarray,
    qmin: np.ndarray,
    qmax: np.ndarray,
    qdot_max: np.ndarray,
    qdd_max: np.ndarray,
    dt: float,
    viability: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
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
    flags = np.empty(n)

    for i in range(n):
        lb, ub, flag = compute_joint_acceleration_bounds(
            q[i], qd[i],
            qmin[i], qmax[i],
            qdot_max[i] if qdot_max.size>1 else float(qdot_max),
            qdd_max[i]  if qdd_max.size>1  else float(qdd_max),
            dt,
            viability
        )
        qdd_lb[i] = lb
        qdd_ub[i] = ub
        flags[i] = flag

    return qdd_lb, qdd_ub, flags
