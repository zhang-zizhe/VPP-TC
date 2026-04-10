#!/usr/bin/env python3
"""Dual-arm OpenArm simulation with VPP-TC inter-arm collision avoidance.

Each arm's end-effector follows a stable limit-cycle dynamical system
(Hopf oscillator).  The two limit cycles overlap in the shared workspace,
forcing the arms to repeatedly cross paths and stress-test the
TransformerGamma collision avoidance.

Usage
-----
    python scripts/simulate_dual_openarm.py
    python scripts/simulate_dual_openarm.py --duration 20 --lc-radius 0.15 --lc-omega 3.0
"""

import argparse
import math
import os
import re
import sys
import time

import cvxpy as cp
import numpy as np
import pandas as pd
import pybullet as p
import pybullet_data
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, os.pardir)
sys.path.insert(0, os.path.abspath(_PROJECT_ROOT))

from vpptc.model import TransformerGamma
from vpptc.safety import compute_joint_acceleration_bounds_vec
from vpptc.utils_openarm import (
    DUAL_ACCELERATION_LIMITS,
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
        description="Dual-arm OpenArm VPP-TC simulation (inter-arm collision avoidance)",
    )
    parser.add_argument("--duration", type=float, default=20,
                        help="Simulation duration in seconds (default: 20)")
    parser.add_argument("--stepsize", type=float, default=1e-3,
                        help="Simulation time step (default: 1e-3)")
    parser.add_argument("--seed", type=int, default=28,
                        help="Random seed (default: 28)")
    parser.add_argument("--lc-center-left", type=float, nargs=3,
                        default=[0.35, 0.03, 0.85],
                        help="Left-arm limit cycle center [x y z] (must be above body z=0.77)")
    parser.add_argument("--lc-center-right", type=float, nargs=3,
                        default=[0.35, -0.03, 0.85],
                        help="Right-arm limit cycle center [x y z]")
    parser.add_argument("--lc-radius", type=float, default=0.1,
                        help="Limit cycle radius in metres (default: 0.1)")
    parser.add_argument("--lc-omega", type=float, default=2.0,
                        help="Limit cycle angular speed (default: 2.0 rad/s)")
    parser.add_argument("--lc-plane-left", type=str, default="xz",
                        choices=["xy", "xz", "yz"],
                        help="Plane for the left-arm limit cycle (default: xz)")
    parser.add_argument("--lc-plane-right", type=str, default="xy",
                        choices=["xy", "xz", "yz"],
                        help="Plane for the right-arm limit cycle (default: xy)")
    parser.add_argument("--gamma-threshold", type=float, default=5,
                        help="Gamma threshold for collision avoidance (gamma<thresh=danger, higher=more conservative)")
    parser.add_argument("--alpha", type=float, default=1e-3,
                        help="Regularisation weight in the QP")
    parser.add_argument("--model-path", type=str,
                        default=os.path.join(
                            _PROJECT_ROOT, "assets", "models",
                            "transformer_gamma_dual_openarm.pt"),
                        help="Path to the dual-arm TransformerGamma weights")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(_PROJECT_ROOT, "output"),
                        help="Directory for result CSV files")
    parser.add_argument("--lc-kd", type=float, default=200,
                        help="DS velocity tracking gain (default: 200)")
    parser.add_argument("--lc-kpos", type=float, default=80.0,
                        help="Position correction gain toward the cycle (default: 80)")
    parser.add_argument("--lc-alpha", type=float, default=20.0,
                        help="Hopf radial convergence rate (default: 20)")
    parser.add_argument("--lc-kperp", type=float, default=20.0,
                        help="Out-of-plane restoring stiffness (default: 20)")
    return parser.parse_args()


# ======================================================================
# Helpers
# ======================================================================

_PLANE_AXES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}
MAX_DS_SPEED = 0.5


def limit_cycle_ds(pos, center, radius, omega, plane="xy",
                   alpha=20.0, k_perp=20.0):
    """Hopf-oscillator stable limit cycle in an arbitrary 2-D plane."""
    i1, i2, i3 = _PLANE_AXES[plane]
    d1 = pos[i1] - center[i1]
    d2 = pos[i2] - center[i2]
    rho_sq = d1 ** 2 + d2 ** 2
    radial = alpha * (radius ** 2 - rho_sq)
    vel = np.zeros(3)
    vel[i1] = radial * d1 - omega * d2
    vel[i2] = radial * d2 + omega * d1
    vel[i3] = -k_perp * (pos[i3] - center[i3])
    speed = np.linalg.norm(vel)
    if speed > MAX_DS_SPEED:
        vel *= MAX_DS_SPEED / speed
    return vel


def compute_ds_force(end_pos, end_vel, center, radius, omega, plane="xy",
                     k_d=100.0, k_pos=80.0, alpha=20.0, k_perp=20.0):
    """Velocity-tracking + position-correction force for the limit cycle DS."""
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
    """Distance from *pos* to the limit-cycle circle in the given plane."""
    i1, i2, _ = _PLANE_AXES[plane]
    d = np.array([pos[i1] - center[i1], pos[i2] - center[i2]])
    return abs(np.linalg.norm(d) - radius)


def draw_circle(center, radius, color, plane="xy", n_seg=50):
    """Visualise a limit cycle as a circle of debug lines in PyBullet."""
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
    """Return the link index whose child-link name matches *name*."""
    for i in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, i)[12].decode("utf-8") == name:
            return i
    raise ValueError(f"Link {name!r} not found")


def compute_gamma_and_grad_dual(model, q_14, qd_14, threshold, device):
    """Evaluate dual-arm Gamma and, if below threshold, its gradient.

    Raw input is fed directly to the model (no normalisation).
    Gradient is w.r.t. the raw [q, qd] input.
    """
    q_t = torch.tensor(q_14, dtype=torch.float32).unsqueeze(0)
    qd_t = torch.tensor(qd_14, dtype=torch.float32).unsqueeze(0)
    x_raw = torch.cat([q_t, qd_t], dim=1).to(device)
    x_raw.requires_grad_(True)
    _, gamma = model(x_raw)
    gamma_val = gamma.item()
    if gamma_val < threshold:
        gamma.backward()
        return gamma_val, x_raw.grad.squeeze(0).cpu().numpy()
    return gamma_val, None


# ======================================================================
# Main
# ======================================================================

def main():
    args = get_args()
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Hardware limits from utils_openarm (left + right = 14 elements)
    acc_max = np.array([a[1] for a in DUAL_ACCELERATION_LIMITS], dtype=np.float32)
    qd_lim = np.array(DUAL_VELOCITY_LIMITS, dtype=np.float32)
    q_min_hw = np.array(DUAL_Q_MIN)
    q_max_hw = np.array(DUAL_Q_MAX)

    # Limit cycle DS parameters
    lc_center_L = np.array(args.lc_center_left)
    lc_center_R = np.array(args.lc_center_right)
    lc_radius = args.lc_radius
    lc_omega = args.lc_omega
    plane_L = args.lc_plane_left
    plane_R = args.lc_plane_right

    # ----- PyBullet setup -----
    p.connect(p.GUI, options="--width=1920 --height=1080")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.5, cameraYaw=50, cameraPitch=-20,
        cameraTargetPosition=[0.2, -0.2, 0.8])
    p.resetSimulation()
    p.setTimeStep(args.stepsize)
    p.setRealTimeSimulation(0)
    p.setGravity(0, 0, 0)

    # Ground plane
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane = p.loadURDF("plane.urdf", useFixedBase=True)
    p.changeDynamics(plane, -1, restitution=0.95)

    # Dual-arm robot (bimanual URDF with hands)
    urdf_path = os.path.join(
        _PROJECT_ROOT, "assets", "urdf", "openarm_description",
        "urdf", "robot", "openarm_bimanual.urdf")
    robot = p.loadURDF(
        urdf_path, useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
    p.changeDynamics(robot, -1, linearDamping=0, angularDamping=0)

    # Exclude only link5↔link7 collision pairs per arm (permanently overlapping)
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

    # Fix finger joints at 0.01 to avoid finger-finger overlap
    finger_joints = []
    for i in range(p.getNumJoints(robot)):
        if "finger_joint" in p.getJointInfo(robot, i)[1].decode("utf-8"):
            finger_joints.append(i)
            p.resetJointState(robot, i, 0.01)

    # Discover ARM joints only (revolute, matching openarm_*_joint[1-7])
    arm_re = re.compile(r"openarm_(left|right)_joint[1-7]$")
    movable_joints = []
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        jname = info[1].decode("utf-8")
        if info[2] == p.JOINT_REVOLUTE and arm_re.match(jname):
            movable_joints.append(i)
    assert len(movable_joints) == N_DOF, (
        f"Expected {N_DOF} arm DOFs, found {len(movable_joints)}")

    # End-effector link indices (hand TCP)
    left_ee = find_link_index(robot, "openarm_left_hand_tcp")
    right_ee = find_link_index(robot, "openarm_right_hand_tcp")

    # Disable default velocity controller on arm joints
    p.setJointMotorControlArray(
        robot, movable_joints, p.VELOCITY_CONTROL, forces=[0.0] * N_DOF)

    # Initial configuration
    q0 = list(Q_HOME_DUAL)
    for jid, angle in zip(movable_joints, q0):
        p.resetJointState(robot, jid, angle)

    # Draw limit cycle circles
    draw_circle(lc_center_L, lc_radius, color=[0, 0.5, 1], plane=plane_L)
    draw_circle(lc_center_R, lc_radius, color=[1, 0.3, 0], plane=plane_R)
    # Small spheres at centres
    vis_l = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.015, rgbaColor=[0, 0.5, 1, 0.6])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_l,
                      basePosition=lc_center_L.tolist())
    vis_r = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.015, rgbaColor=[1, 0.3, 0, 0.6])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_r,
                      basePosition=lc_center_R.tolist())

    # ----- Load dual-arm Gamma model + normalisation stats -----
    device = torch.device("cpu")
    gamma_model = TransformerGamma(input_dim=N_DOF * 2).to(device)
    gamma_model.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=True))
    gamma_model.eval()

    # ALL movable joints (arm + finger) — needed for PyBullet dynamics calls
    all_movable_joints = []
    for i in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot, i)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            all_movable_joints.append(i)
    n_all_dof = len(all_movable_joints)

    # Map: arm joint index in movable_joints → index in all_movable_joints
    arm_indices_in_all = [all_movable_joints.index(j) for j in movable_joints]

    # All links per arm for inter-arm distance check (including hand/finger)
    left_body_links = []
    right_body_links = []
    for i in range(p.getNumJoints(robot)):
        name = p.getJointInfo(robot, i)[12].decode("utf-8")
        if name.startswith("openarm_left_"):
            left_body_links.append(i)
        elif name.startswith("openarm_right_"):
            right_body_links.append(i)

    # ----- Data loggers -----
    log = {k: [] for k in [
        "time", "collision_dist", "inter_arm_dist", "gamma",
        "lc_dist_left", "lc_dist_right", "runtime_ms",
    ]}

    time.sleep(2)
    wall_start = time.time()
    num_steps = int(args.duration / args.stepsize)
    sim_t = 0.0
    zeros_all = [0.0] * n_all_dof

    for step_i in range(num_steps):
        t0 = time.perf_counter()

        # --- Joint states (14 DOFs) ---
        states = p.getJointStates(robot, movable_joints)
        q = [s[0] for s in states]
        qd = [s[1] for s in states]
        q_np = np.array(q)
        qd_np = np.array(qd)

        # --- FK and EE velocities ---
        ls_left = p.getLinkState(robot, left_ee, computeLinkVelocity=True)
        ls_right = p.getLinkState(robot, right_ee, computeLinkVelocity=True)
        left_pos = np.array(ls_left[0])
        left_vel = np.array(ls_left[6])
        right_pos = np.array(ls_right[0])
        right_vel = np.array(ls_right[6])

        # --- Limit cycle DS forces ---
        dr_L = radial_error(left_pos, lc_center_L, lc_radius, plane_L)
        dr_R = radial_error(right_pos, lc_center_R, lc_radius, plane_R)
        fc_left = compute_ds_force(
            left_pos, left_vel, lc_center_L, lc_radius, lc_omega, plane_L,
            k_d=args.lc_kd, k_pos=args.lc_kpos,
            alpha=args.lc_alpha, k_perp=args.lc_kperp)
        fc_right = compute_ds_force(
            right_pos, right_vel, lc_center_R, lc_radius, -lc_omega, plane_R,
            k_d=args.lc_kd, k_pos=args.lc_kpos,
            alpha=args.lc_alpha, k_perp=args.lc_kperp)

        # --- Viability stopping position ---
        qe = compute_qe(q, qd, acc_limits=DUAL_ACCELERATION_LIMITS)

        # --- Self/inter-arm collision safety ---
        Gamma, grad_gamma = compute_gamma_and_grad_dual(
            gamma_model, q, qd, args.gamma_threshold, device)

        # --- Real-time status ---
        if step_i % 100 == 0:
            print(f"[sim] t={sim_t:.3f}s  Gamma={Gamma:+.4f}"
                  f"  avoidance={'ON' if grad_gamma is not None else 'off'}")

        # --- Acceleration bounds ---
        qdd_lb, qdd_ub = compute_joint_acceleration_bounds_vec(
            q_np, qd_np, q_min_hw, q_max_hw, qd_lim, acc_max,
            dt=0.02, viability=True)

        # --- Full joint states for dynamics calls (arm + finger joints) ---
        all_states = p.getJointStates(robot, all_movable_joints)
        all_pos = [s[0] for s in all_states]
        all_vel = [s[1] for s in all_states]

        # --- Jacobians (each 3 x n_all_dof, then extract arm columns) ---
        J_left_full = np.array(p.calculateJacobian(
            robot, left_ee, [0, 0, 0], all_pos, zeros_all, zeros_all)[0])
        J_right_full = np.array(p.calculateJacobian(
            robot, right_ee, [0, 0, 0], all_pos, zeros_all, zeros_all)[0])
        # Extract only arm joint columns (14 out of n_all_dof)
        J_left = J_left_full[:, arm_indices_in_all]      # (3, 14)
        J_right = J_right_full[:, arm_indices_in_all]     # (3, 14)
        J_stack = np.vstack([J_left, J_right])            # (6, 14)
        fc_stack = np.concatenate([fc_left, fc_right])    # (6,)
        JT_pinv = np.linalg.pinv(J_stack.T)               # (6, 14)

        # --- Dynamics (full DOF, then extract arm submatrix) ---
        M_full = np.array(p.calculateMassMatrix(robot, all_pos))
        tau_id_full = np.array(p.calculateInverseDynamics(
            robot, all_pos, all_vel, zeros_all))
        # Extract arm-joint submatrix/subvector
        ix = np.array(arm_indices_in_all)
        M = M_full[np.ix_(ix, ix)]                        # (14, 14)
        tau_id = tau_id_full[ix]                           # (14,)
        M_inv = np.linalg.inv(M)

        # Ensure feasibility of acceleration bounds
        for idx in range(N_DOF):
            if qdd_lb[idx] > qdd_ub[idx]:
                qdd_lb[idx] = qdd_ub[idx] - 1e-4

        # --- QP controller (14-DOF) ---
        u = cp.Variable(N_DOF)
        objective = (cp.sum_squares(JT_pinv @ u - fc_stack)
                     + args.alpha * cp.sum_squares(u))
        constraints = [
            M_inv @ u >= qdd_lb + M_inv @ tau_id,
            M_inv @ u <= qdd_ub + M_inv @ tau_id,
        ]

        soft = False
        if grad_gamma is not None:
            dt = 0.02
            gq = grad_gamma[:N_DOF]
            gqd = grad_gamma[N_DOF:]
            g_eff = 0.5 * gq * dt ** 2 + gqd * dt
            c_const = gq.dot(qd_np) * dt
            eps = 4e-1
            constraints.append(
                g_eff @ M_inv @ u >= eps - c_const + g_eff @ M_inv @ tau_id
            )
            prob = cp.Problem(cp.Minimize(objective), constraints)
            try:
                prob.solve(solver=cp.OSQP)
            except cp.SolverError:
                soft = True

            while prob.status != cp.OPTIMAL and eps > 1e-3:
                eps -= 1e-1
                constraints[-1] = (
                    g_eff @ M_inv @ u >= eps - c_const + g_eff @ M_inv @ tau_id
                )
                prob = cp.Problem(cp.Minimize(objective), constraints)
                prob.solve(solver=cp.OSQP)

            if eps < 1e-3:
                soft = True

            if soft:
                qdd_cmd = np.where(g_eff > 0, qdd_ub, qdd_lb)
                # Build full-DOF acceleration array (zeros for finger joints)
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
            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve(solver=cp.OSQP)

        if not soft:
            tau_cmd = np.array(u.value)
            p.setJointMotorControlArray(
                robot, movable_joints, p.TORQUE_CONTROL,
                forces=tau_cmd.tolist())
            p.stepSimulation()

        sim_t += args.stepsize
        t1 = time.perf_counter()

        # --- Inter-arm closest distance ---
        inter_dist = math.inf
        for l1 in left_body_links:
            for l2 in right_body_links:
                pts = p.getClosestPoints(robot, robot, 2.0, l1, l2)
                if pts:
                    inter_dist = min(inter_dist, min(pt[8] for pt in pts))

        # --- Collision detection ---
        contacts = p.getContactPoints(bodyA=robot, bodyB=robot)
        collision = len(contacts) > 0
        collision_dist = min(c[8] for c in contacts) if contacts else inter_dist

        # --- Log ---
        log["time"].append(sim_t)
        log["collision_dist"].append(collision_dist)
        log["inter_arm_dist"].append(inter_dist)
        log["gamma"].append(Gamma)
        log["lc_dist_left"].append(dr_L)
        log["lc_dist_right"].append(dr_R)
        log["runtime_ms"].append((t1 - t0) * 1e3)

        if collision:
            print(f"[sim] t={sim_t:.3f}s  COLLISION  dist={collision_dist:.5f}"
                  f"  inter_arm={inter_dist:.5f}  Gamma={Gamma:+.4f}")
            break

        time.sleep(args.stepsize / 3)

    # --- Summary ---
    elapsed = time.time() - wall_start
    runtimes = np.array(log["runtime_ms"][1:])
    print(f"\n{'=' * 50}")
    print(f"Simulation finished in {elapsed:.2f}s wall-clock")
    print(f"  Steps         : {len(log['time'])}")
    if len(runtimes) > 0:
        print(f"  Mean step     : {runtimes.mean():.3f} ms")
        print(f"  Median step   : {np.median(runtimes):.3f} ms")
        print(f"  95th pct      : {np.percentile(runtimes, 95):.3f} ms")
        print(f"  Max step      : {runtimes.max():.3f} ms")

    # --- Save ---
    df = pd.DataFrame(log)
    csv_path = os.path.join(args.output_dir, f"openarm_dual_run_{int(time.time())}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
