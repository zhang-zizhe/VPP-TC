#!/usr/bin/env python3
"""Dual-arm OpenArm simulation -- Plan A (GammaRegressor single-head).

Differences vs the legacy simulate_dual_openarm_dist.py (which used
TransformerGamma dual-head):

  * Loads `GammaRegressor` via the new checkpoint format
    `{"state_dict": ..., "config": ...}`.
  * Model returns scalar gamma directly (no tuple).
  * Gamma is signed distance in metres; classification is `gamma < margin`.
  * Default margin bumped to 20 mm because the current model has
    bias=+4.4 mm (over-predicts) and rec@5mm is only 82%.  Margin needs
    to absorb both effects.

The QP / DS / dynamics path is identical to the original.

Usage
-----
    python scripts/simulate/simulate_dual_openarm_dist.py
    python scripts/simulate/simulate_dual_openarm_dist.py --duration 30 \
        --gamma-threshold 0.020
"""

import argparse
import math
import os
import re
import sys
import time

# --- BLAS thread cap (MUST be set before numpy/cvxpy/torch import OpenBLAS) ---
# This sim runs many tiny per-step linear-algebra ops (per-step OSQP solve,
# inverse dynamics, small matrices).  OpenBLAS defaults to one thread per core
# (16 here) and burns nearly all its time on thread sync -- measured ~8x slower
# wall-clock with sys-time exploding.  Windows MKL doesn't oversubscribe tiny
# ops, which is why the Windows run was much faster.  setdefault => CLI/env can
# still override (e.g. OPENBLAS_NUM_THREADS=4 python3 ...).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import cvxpy as cp
import numpy as np
import pandas as pd
import pybullet as p
import pybullet_data
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
sys.path.insert(0, _PROJECT_ROOT)

from vpptc.model import GammaRegressor
from vpptc.safety import compute_joint_acceleration_bounds_vec
from vpptc.utils_openarm import (
    DUAL_ACCELERATION_LIMITS,
    DUAL_POS_LIMITS,
    DUAL_Q_MAX,
    DUAL_Q_MIN,
    DUAL_VELOCITY_LIMITS,
    N_DOF,
    Q_HOME_DUAL,
    compute_qe,
)


# ======================================================================
# Argument parsing
# ======================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Dual-arm OpenArm Plan A simulation (GammaRegressor)",
    )
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("--stepsize", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--rand-init", action="store_true",
                        help="Start from a RANDOM collision-free joint config "
                             "(perturb home, seeded by --seed) instead of the "
                             "fixed Q_HOME_DUAL home pose. Same logic as the "
                             "classifier sim, for matched robustness testing.")
    parser.add_argument("--lc-center-left",  type=float, nargs=3,
                        default=[0.35, 0.03, 0.85])
    parser.add_argument("--lc-center-right", type=float, nargs=3,
                        default=[0.35, -0.03, 0.85])
    parser.add_argument("--lc-radius", type=float, default=0.1)
    parser.add_argument("--lc-omega",  type=float, default=2.0)
    parser.add_argument("--lc-plane-left",  type=str, default="xz",
                        choices=["xy", "xz", "yz"])
    parser.add_argument("--lc-plane-right", type=str, default="xy",
                        choices=["xy", "xz", "yz"])
    parser.add_argument("--gamma-threshold", type=float, default=0.020,
                        help="Trigger avoidance when predicted dist < this. "
                             "Default 20mm absorbs model bias (+4mm) and "
                             "low recall at 5mm.  Increase for safer / "
                             "more conservative behaviour.")
    parser.add_argument("--sca-eps",       type=float, default=1e-3)
    parser.add_argument("--sca-eps-decay", type=float, default=2e-4)
    parser.add_argument("--sca-eps-floor", type=float, default=1e-5)
    parser.add_argument("--alpha", type=float, default=1e-3,
                        help="QP regularisation weight")
    parser.add_argument("--model-path", type=str,
                        default=os.path.join(
                            _PROJECT_ROOT, "assets", "models",
                            "gamma_regressor_openarm.pt"))
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(_PROJECT_ROOT, "output"))
    parser.add_argument("--lc-kd",     type=float, default=200)
    parser.add_argument("--lc-kpos",   type=float, default=80.0)
    parser.add_argument("--lc-alpha",  type=float, default=20.0)
    parser.add_argument("--lc-kperp",  type=float, default=20.0)
    parser.add_argument("--no-gui", action="store_true",
                        help="Headless (DIRECT) mode")
    return parser.parse_args()


# ======================================================================
# Helpers (identical to legacy script)
# ======================================================================

_PLANE_AXES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}
MAX_DS_SPEED = 0.5


def limit_cycle_ds(pos, center, radius, omega, plane="xy",
                   alpha=20.0, k_perp=20.0):
    i1, i2, i3 = _PLANE_AXES[plane]
    d1 = pos[i1] - center[i1]
    d2 = pos[i2] - center[i2]
    rho_sq = d1 ** 2 + d2 ** 2
    radial = alpha * (radius ** 2 - rho_sq)
    vel = np.zeros(3)
    vel[i1] = radial * d1 - omega * d2
    vel[i2] = radial * d2 + omega * d1
    vel[i3] = -k_perp * (pos[i3] - center[i3])
    s = np.linalg.norm(vel)
    if s > MAX_DS_SPEED:
        vel *= MAX_DS_SPEED / s
    return vel


def compute_ds_force(end_pos, end_vel, center, radius, omega, plane="xy",
                     k_d=100.0, k_pos=80.0, alpha=20.0, k_perp=20.0):
    i1, i2, i3 = _PLANE_AXES[plane]
    v_des = limit_cycle_ds(end_pos, center, radius, omega, plane,
                           alpha=alpha, k_perp=k_perp)
    f_pos = np.zeros(3)
    d = np.array([end_pos[i1] - center[i1], end_pos[i2] - center[i2]])
    rho = np.linalg.norm(d)
    if rho > 1e-8:
        direction = d / rho
        f_pos[i1] = -k_pos * (rho - radius) * direction[0]
        f_pos[i2] = -k_pos * (rho - radius) * direction[1]
    f_pos[i3] = -k_pos * (end_pos[i3] - center[i3])
    return k_d * (v_des - end_vel) + f_pos


def radial_error(pos, center, radius, plane="xy"):
    i1, i2, _ = _PLANE_AXES[plane]
    d = np.array([pos[i1] - center[i1], pos[i2] - center[i2]])
    return abs(np.linalg.norm(d) - radius)


def draw_circle(center, radius, color, plane="xy", n_seg=50):
    i1, i2, _ = _PLANE_AXES[plane]
    for i in range(n_seg):
        a1 = 2 * math.pi * i / n_seg
        a2 = 2 * math.pi * (i + 1) / n_seg
        pt1, pt2 = list(center), list(center)
        pt1[i1] = center[i1] + radius * math.cos(a1)
        pt1[i2] = center[i2] + radius * math.sin(a1)
        pt2[i1] = center[i1] + radius * math.cos(a2)
        pt2[i2] = center[i2] + radius * math.sin(a2)
        p.addUserDebugLine(pt1, pt2, color, lineWidth=2, lifeTime=0)


def find_link_index(robot_id, name):
    for i in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, i)[12].decode("utf-8") == name:
            return i
    raise ValueError(f"Link {name!r} not found")


# ---- NEW: GammaRegressor gradient computation ----------------------------

def compute_gamma_and_grad(model, q_14, qd_14, threshold, device):
    """GammaRegressor: scalar gamma + (optional) gradient w.r.t. [q, qd].

    Differs from the legacy dual-head version: model returns a single
    tensor (not a tuple).  Gradient only computed when threshold is
    exceeded, to avoid the autograd overhead in the safe-region case.
    """
    q_t = torch.tensor(q_14, dtype=torch.float32).unsqueeze(0)
    qd_t = torch.tensor(qd_14, dtype=torch.float32).unsqueeze(0)
    x = torch.cat([q_t, qd_t], dim=1).to(device)
    x.requires_grad_(True)
    gamma = model(x)
    gamma_val = float(gamma.item())
    if gamma_val < threshold:
        gamma.backward()
        return gamma_val, x.grad.squeeze(0).cpu().numpy()
    return gamma_val, None


# ======================================================================
# Main
# ======================================================================

def main():
    args = get_args()
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    acc_max  = np.array([a[1] for a in DUAL_ACCELERATION_LIMITS], dtype=np.float32)
    qd_lim   = np.array(DUAL_VELOCITY_LIMITS, dtype=np.float32)
    q_min_hw = np.array(DUAL_Q_MIN)
    q_max_hw = np.array(DUAL_Q_MAX)

    lc_center_L = np.array(args.lc_center_left)
    lc_center_R = np.array(args.lc_center_right)

    # ----- PyBullet setup -----
    mode = p.DIRECT if args.no_gui else p.GUI
    if args.no_gui:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI, options="--width=1920 --height=1080")
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5, cameraYaw=50, cameraPitch=-20,
            cameraTargetPosition=[0.2, -0.2, 0.8])
    p.resetSimulation()
    p.setTimeStep(args.stepsize)
    p.setRealTimeSimulation(0)
    p.setGravity(0, 0, 0)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane = p.loadURDF("plane.urdf", useFixedBase=True)
    p.changeDynamics(plane, -1, restitution=0.95)

    urdf_path = os.path.join(
        _PROJECT_ROOT, "assets", "urdf", "openarm_description",
        "urdf", "robot", "openarm_bimanual.urdf")
    robot = p.loadURDF(
        urdf_path, useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
    p.changeDynamics(robot, -1, linearDamping=0, angularDamping=0)

    # Exclude link5↔link7 within each arm (permanently overlapping URDF artefact)
    link5s, link7s = [], []
    for i in range(p.getNumJoints(robot)):
        name = p.getJointInfo(robot, i)[12].decode("utf-8")
        if name.endswith("_link5"):
            link5s.append(i)
        elif name.endswith("_link7"):
            link7s.append(i)
    for l5 in link5s:
        for l7 in link7s:
            if abs(l5 - l7) < 5:
                p.setCollisionFilterPair(robot, robot, l5, l7, enableCollision=0)

    # Fingers fixed
    for i in range(p.getNumJoints(robot)):
        if "finger_joint" in p.getJointInfo(robot, i)[1].decode("utf-8"):
            p.resetJointState(robot, i, 0.01)

    arm_re = re.compile(r"openarm_(left|right)_joint[1-7]$")
    movable_joints = []
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        if info[2] == p.JOINT_REVOLUTE and arm_re.match(info[1].decode("utf-8")):
            movable_joints.append(i)
    assert len(movable_joints) == N_DOF

    left_ee  = find_link_index(robot, "openarm_left_hand_tcp")
    right_ee = find_link_index(robot, "openarm_right_hand_tcp")

    p.setJointMotorControlArray(
        robot, movable_joints, p.VELOCITY_CONTROL, forces=[0.0] * N_DOF)

    if args.rand_init:
        # Random collision-free start: perturb home by per-joint noise (same
        # logic as the classifier sim, for matched testing). getContactPoints
        # respects collision filters & reports only real contacts.
        rng = np.random.default_rng(args.seed)
        home = np.array(Q_HOME_DUAL)
        q0 = None
        for _try in range(400):
            scale = 0.9 * (0.97 ** _try)
            cand = np.clip(home + rng.uniform(-scale, scale, size=N_DOF),
                           q_min_hw, q_max_hw)
            for jid, angle in zip(movable_joints, cand):
                p.resetJointState(robot, jid, float(angle))
            p.performCollisionDetection()
            contacts = p.getContactPoints(bodyA=robot, bodyB=robot)
            if not any(c[8] < -5e-4 for c in contacts):
                q0 = list(cand)
                break
        if q0 is None:
            print("  [rand-init] WARN: no collision-free sample, using home")
            q0 = list(Q_HOME_DUAL)
        else:
            print(f"  [rand-init] seed={args.seed} -> collision-free start "
                  f"(try {_try+1}, noise +-{np.degrees(scale):.0f}deg)")
    else:
        q0 = list(Q_HOME_DUAL)
    for jid, angle in zip(movable_joints, q0):
        p.resetJointState(robot, jid, angle)

    if not args.no_gui:
        draw_circle(lc_center_L, args.lc_radius,
                    color=[0, 0.5, 1], plane=args.lc_plane_left)
        draw_circle(lc_center_R, args.lc_radius,
                    color=[1, 0.3, 0], plane=args.lc_plane_right)
        vis_l = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.015, rgbaColor=[0, 0.5, 1, 0.6])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_l,
                          basePosition=lc_center_L.tolist())
        vis_r = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.015, rgbaColor=[1, 0.3, 0, 0.6])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_r,
                          basePosition=lc_center_R.tolist())

    # ----- Load GammaRegressor (Plan A) -----
    device = torch.device("cpu")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    cfg.setdefault("output_mode", "tanh")  # old checkpoints were tanh
    gamma_model = GammaRegressor(**cfg).to(device).eval()
    gamma_model.load_state_dict(ckpt["state_dict"])
    print(f"  [model] GammaRegressor  config={cfg}")
    print(f"  [sca]   threshold={args.gamma_threshold*1000:.1f}mm  "
          f"eps_start={args.sca_eps}  eps_decay={args.sca_eps_decay}  "
          f"eps_floor={args.sca_eps_floor}")

    # All movable joints (arms + fingers) for dynamics calls
    all_movable_joints = []
    for i in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot, i)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            all_movable_joints.append(i)
    n_all_dof = len(all_movable_joints)
    arm_indices_in_all = [all_movable_joints.index(j) for j in movable_joints]

    left_body_links, right_body_links = [], []
    for i in range(p.getNumJoints(robot)):
        name = p.getJointInfo(robot, i)[12].decode("utf-8")
        if name.startswith("openarm_left_"):
            left_body_links.append(i)
        elif name.startswith("openarm_right_"):
            right_body_links.append(i)

    log = {k: [] for k in [
        "time", "collision_dist", "inter_arm_dist", "gamma",
        "lc_dist_left", "lc_dist_right", "runtime_ms", "avoidance_on",
    ]}

    if not args.no_gui:
        time.sleep(2)
    wall_start = time.time()
    num_steps = int(args.duration / args.stepsize)
    sim_t = 0.0
    zeros_all = [0.0] * n_all_dof
    n_avoidance = 0
    n_soft = 0

    for step_i in range(num_steps):
        t0 = time.perf_counter()

        states = p.getJointStates(robot, movable_joints)
        q  = [s[0] for s in states]
        qd = [s[1] for s in states]
        q_np  = np.array(q)
        qd_np = np.array(qd)

        ls_left  = p.getLinkState(robot, left_ee,  computeLinkVelocity=True)
        ls_right = p.getLinkState(robot, right_ee, computeLinkVelocity=True)
        left_pos,  left_vel  = np.array(ls_left[0]),  np.array(ls_left[6])
        right_pos, right_vel = np.array(ls_right[0]), np.array(ls_right[6])

        dr_L = radial_error(left_pos,  lc_center_L, args.lc_radius, args.lc_plane_left)
        dr_R = radial_error(right_pos, lc_center_R, args.lc_radius, args.lc_plane_right)
        fc_left  = compute_ds_force(
            left_pos, left_vel, lc_center_L, args.lc_radius,
            args.lc_omega, args.lc_plane_left,
            k_d=args.lc_kd, k_pos=args.lc_kpos,
            alpha=args.lc_alpha, k_perp=args.lc_kperp)
        fc_right = compute_ds_force(
            right_pos, right_vel, lc_center_R, args.lc_radius,
            -args.lc_omega, args.lc_plane_right,
            k_d=args.lc_kd, k_pos=args.lc_kpos,
            alpha=args.lc_alpha, k_perp=args.lc_kperp)

        qe = compute_qe(q, qd, acc_limits=DUAL_ACCELERATION_LIMITS,
                        pos_limits=DUAL_POS_LIMITS)

        # ---- Plan A: GammaRegressor predicts signed distance directly ----
        Gamma, grad_gamma = compute_gamma_and_grad(
            gamma_model, q, qd, args.gamma_threshold, device)

        if step_i % 100 == 0:
            print(f"[sim] t={sim_t:6.3f}s  Gamma={Gamma*1000:+7.2f}mm  "
                  f"avoidance={'ON ' if grad_gamma is not None else 'off'}")

        qdd_lb, qdd_ub = compute_joint_acceleration_bounds_vec(
            q_np, qd_np, q_min_hw, q_max_hw, qd_lim, acc_max,
            dt=0.02, viability=True)

        all_states = p.getJointStates(robot, all_movable_joints)
        all_pos = [s[0] for s in all_states]
        all_vel = [s[1] for s in all_states]

        J_left_full  = np.array(p.calculateJacobian(
            robot, left_ee,  [0, 0, 0], all_pos, zeros_all, zeros_all)[0])
        J_right_full = np.array(p.calculateJacobian(
            robot, right_ee, [0, 0, 0], all_pos, zeros_all, zeros_all)[0])
        J_left  = J_left_full[:,  arm_indices_in_all]
        J_right = J_right_full[:, arm_indices_in_all]
        J_stack = np.vstack([J_left, J_right])
        fc_stack = np.concatenate([fc_left, fc_right])
        JT_pinv  = np.linalg.pinv(J_stack.T)

        M_full = np.array(p.calculateMassMatrix(robot, all_pos))
        tau_id_full = np.array(p.calculateInverseDynamics(
            robot, all_pos, all_vel, zeros_all))
        ix = np.array(arm_indices_in_all)
        M = M_full[np.ix_(ix, ix)]
        tau_id = tau_id_full[ix]
        M_inv = np.linalg.inv(M)

        for idx in range(N_DOF):
            if qdd_lb[idx] > qdd_ub[idx]:
                qdd_lb[idx] = qdd_ub[idx] - 1e-4

        u = cp.Variable(N_DOF)
        objective = (cp.sum_squares(JT_pinv @ u - fc_stack)
                     + args.alpha * cp.sum_squares(u))
        constraints = [
            M_inv @ u >= qdd_lb + M_inv @ tau_id,
            M_inv @ u <= qdd_ub + M_inv @ tau_id,
        ]

        soft = False
        if grad_gamma is not None:
            n_avoidance += 1
            dt = 0.02
            gq  = grad_gamma[:N_DOF]
            gqd = grad_gamma[N_DOF:]
            g_eff = 0.5 * gq * dt ** 2 + gqd * dt
            c_const = gq.dot(qd_np) * dt
            eps = args.sca_eps
            constraints.append(
                g_eff @ M_inv @ u >= eps - c_const + g_eff @ M_inv @ tau_id
            )
            prob = cp.Problem(cp.Minimize(objective), constraints)
            try:
                prob.solve(solver=cp.OSQP, max_iter=20000, eps_abs=1e-5, eps_rel=1e-5)
            except cp.SolverError:
                soft = True
                print(f"[sim] t={sim_t:6.3f}s  OSQP SolverError "
                      f"(avoidance solve, eps={eps:.3f}) -> soft")

            while (not soft) and prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) and eps > args.sca_eps_floor:
                eps -= args.sca_eps_decay
                constraints[-1] = (
                    g_eff @ M_inv @ u >= eps - c_const + g_eff @ M_inv @ tau_id
                )
                prob = cp.Problem(cp.Minimize(objective), constraints)
                try:
                    prob.solve(solver=cp.OSQP, max_iter=20000, eps_abs=1e-5, eps_rel=1e-5)
                except cp.SolverError:
                    soft = True
                    print(f"[sim] t={sim_t:6.3f}s  OSQP SolverError "
                          f"(SCA retry, eps={eps:.3f}) -> soft")
                    break

            if soft or eps < args.sca_eps_floor or prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                soft = True

            if soft:
                n_soft += 1
                qdd_cmd = np.where(g_eff > 0, qdd_ub, qdd_lb)
                qdd_full = [0.0] * n_all_dof
                for k, idx in enumerate(arm_indices_in_all):
                    qdd_full[idx] = qdd_cmd[k]
                tau_full = list(p.calculateInverseDynamics(
                    robot, all_pos, all_vel, qdd_full))
                tau_arm = [tau_full[idx] for idx in arm_indices_in_all]
                p.setJointMotorControlArray(
                    robot, movable_joints, p.TORQUE_CONTROL, forces=tau_arm)
                p.stepSimulation()
        else:
            # Safe region (grad_gamma is None): plain QP, only accel-bound
            # constraints.  Mirror the avoidance branch's QP-failure handling
            # (try/except SolverError + non-OPTIMAL -> soft).  The upstream
            # VPP-TC repo left this branch unguarded; OSQP can still throw on a
            # numerically borderline solve, which crashes the loop.
            prob = cp.Problem(cp.Minimize(objective), constraints)
            try:
                prob.solve(solver=cp.OSQP, max_iter=20000, eps_abs=1e-5, eps_rel=1e-5)
            except cp.SolverError:
                soft = True
                print(f"[sim] t={sim_t:6.3f}s  OSQP SolverError "
                      f"(safe-region solve) -> soft")
            if not soft and prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                soft = True
            if soft:
                # No gamma gradient to follow here, so the consistent soft
                # fallback is zero-acceleration hold: tau = ID(q, qd, qdd=0)
                # (gravity/coriolis compensation), then step -- same "fall back
                # to an inverse-dynamics torque + step" shape as the avoidance
                # branch's bang-bang path.
                n_soft += 1
                tau_full = list(p.calculateInverseDynamics(
                    robot, all_pos, all_vel, [0.0] * n_all_dof))
                tau_arm = [tau_full[idx] for idx in arm_indices_in_all]
                p.setJointMotorControlArray(
                    robot, movable_joints, p.TORQUE_CONTROL, forces=tau_arm)
                p.stepSimulation()

        if not soft:
            tau_cmd = np.array(u.value)
            p.setJointMotorControlArray(
                robot, movable_joints, p.TORQUE_CONTROL,
                forces=tau_cmd.tolist())
            p.stepSimulation()

        sim_t += args.stepsize
        t1 = time.perf_counter()

        # Inter-arm closest distance (ground truth)
        inter_dist = math.inf
        for l1 in left_body_links:
            for l2 in right_body_links:
                pts = p.getClosestPoints(robot, robot, 2.0, l1, l2)
                if pts:
                    inter_dist = min(inter_dist, min(pt[8] for pt in pts))

        contacts_all = p.getContactPoints(bodyA=robot, bodyB=robot)
        contacts = [c for c in contacts_all if c[8] < 0]
        collision = len(contacts) > 0
        collision_dist = min(c[8] for c in contacts) if contacts else inter_dist

        log["time"].append(sim_t)
        log["collision_dist"].append(collision_dist)
        log["inter_arm_dist"].append(inter_dist)
        log["gamma"].append(Gamma)
        log["lc_dist_left"].append(dr_L)
        log["lc_dist_right"].append(dr_R)
        log["runtime_ms"].append((t1 - t0) * 1e3)
        log["avoidance_on"].append(1 if grad_gamma is not None else 0)

        if collision:
            _link_name = {-1: "base"}
            for _i in range(p.getNumJoints(robot)):
                _link_name[_i] = p.getJointInfo(robot, _i)[12].decode("utf-8")
            from collections import Counter as _Counter
            pair_ctr = _Counter()
            for c in contacts:
                a, b = c[3], c[4]
                if a == b:
                    continue
                pair_ctr[(min(a, b), max(a, b))] += 1
            print(f"\n[sim] t={sim_t:.3f}s  COLLISION  "
                  f"dist={collision_dist*1000:+.2f}mm  "
                  f"inter_arm={inter_dist*1000:+.2f}mm  "
                  f"Gamma_pred={Gamma*1000:+.2f}mm")
            print(f"      true_dist - pred_gamma = "
                  f"{(inter_dist - Gamma)*1000:+.2f}mm  "
                  f"(positive = model over-predicted = lied about safety)")
            print(f"      Contact pairs:")
            for (a, b), n in pair_ctr.most_common():
                depths = [c[8] for c in contacts
                          if (min(c[3], c[4]), max(c[3], c[4])) == (a, b)]
                print(f"        x{n:3d}  {_link_name.get(a, a):37s} "
                      f"<-> {_link_name.get(b, b):37s}  "
                      f"min={min(depths)*1000:+.2f}mm")
            break

        if not args.no_gui:
            time.sleep(args.stepsize / 3)

    # --- Summary ---
    elapsed = time.time() - wall_start
    runtimes = np.array(log["runtime_ms"][1:])
    n_steps = len(log["time"])
    print(f"\n{'=' * 60}")
    print(f"Simulation finished in {elapsed:.2f}s wall-clock")
    print(f"  Steps           : {n_steps}")
    print(f"  Avoidance on    : {n_avoidance}/{n_steps}  "
          f"({100*n_avoidance/max(1,n_steps):.1f}%)")
    print(f"  Soft fallback   : {n_soft}/{max(1,n_avoidance)}  "
          f"(QP infeasible)")
    if len(runtimes) > 0:
        print(f"  Mean step       : {runtimes.mean():.3f} ms")
        print(f"  Median step     : {np.median(runtimes):.3f} ms")
        print(f"  95th pct        : {np.percentile(runtimes, 95):.3f} ms")
        print(f"  Max step        : {runtimes.max():.3f} ms")

    # Compare predicted Gamma vs true inter-arm distance
    g = np.array(log["gamma"])
    d = np.array([x for x in log["inter_arm_dist"]
                  if x != math.inf and not math.isinf(x)])
    if len(d) > 100 and len(g) == len(log["inter_arm_dist"]):
        valid = np.isfinite(log["inter_arm_dist"])
        d_v = np.array(log["inter_arm_dist"])[valid]
        g_v = g[valid]
        err = g_v - d_v
        print(f"\n  Predicted vs true (inter-arm only):")
        print(f"    bias        : {err.mean()*1000:+.2f}mm  "
              f"({'over' if err.mean()>0 else 'under'}-predicts)")
        print(f"    RMSE        : {np.sqrt((err**2).mean())*1000:.2f}mm")
        print(f"    min true d  : {d_v.min()*1000:+.2f}mm  "
              f"(closest the arms came)")


if __name__ == "__main__":
    main()
