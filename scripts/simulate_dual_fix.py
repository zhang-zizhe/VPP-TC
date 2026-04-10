#!/usr/bin/env python3
"""Dual-arm Panda simulation with VPP-TC inter-arm collision avoidance.

Drives two Panda arms toward fixed target points (crossing) to test whether
the TransformerGamma model can prevent inter-arm collisions.  No external
obstacles are present; only self/inter-arm collision avoidance is active.

Usage
-----
    python scripts/simulate_dual0
    python scripts/simulate_dual0 --duration 10 --target-left 0.3 -0.3 0.5
"""

import argparse
import math
import os
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
from vpptc.utils import JOINT_ACCELERATION_LIMITS, compute_qe

DUAL_ACC_LIMITS = JOINT_ACCELERATION_LIMITS + JOINT_ACCELERATION_LIMITS
Q_HOME = [0, -0.785, 0, -2.356, 0, 1.571, 0]
N_DOF = 14


# ======================================================================
# Argument parsing
# ======================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Dual-arm VPP-TC simulation (fixed-target, inter-arm collision avoidance)",
    )
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Simulation duration in seconds")
    parser.add_argument("--stepsize", type=float, default=2e-3,
                        help="Simulation time step (default: 2e-3)")
    parser.add_argument("--seed", type=int, default=28,
                        help="Random seed (default: 28)")
    parser.add_argument("--target-left", type=float, nargs=3,
                        default=[0.3, -0.15, 0.3],
                        help="Target for the LEFT arm (pushed right) [x y z]")
    parser.add_argument("--target-right", type=float, nargs=3,
                        default=[0.2, 0.15, 0.3],
                        help="Target for the RIGHT arm (pushed left) [x y z]")
    parser.add_argument("--gamma-threshold", type=float, default=4,
                        help="Gamma threshold for collision avoidance")
    parser.add_argument("--alpha", type=float, default=1e-2,
                        help="Regularisation weight in the QP")
    parser.add_argument("--model-path", type=str,
                        default=os.path.join(
                            _PROJECT_ROOT, "assets", "models",
                            "transformer_gamma_dual_d0.6.pt"),
                        help="Path to the dual-arm TransformerGamma weights")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(_PROJECT_ROOT, "output"),
                        help="Directory for result CSV files")
    return parser.parse_args()


# ======================================================================
# Helpers
# ======================================================================

def compute_impedance_force(end_pos, end_vel, target,
                            l1=5, l2=100, l3=100, k_pos=50.0):
    """Cartesian impedance controller producing a desired end-effector force."""
    fx = -k_pos * (end_pos - target)
    norm = np.linalg.norm(fx)
    if norm < 1e-8:
        return np.zeros(3)
    e1 = fx / norm
    e2 = np.array([1.0, 0.0, 0.0]) - np.dot([1, 0, 0], e1) * e1
    n2 = np.linalg.norm(e2)
    if n2 < 1e-8:
        e2 = np.array([0.0, 1.0, 0.0]) - np.dot([0, 1, 0], e1) * e1
        n2 = np.linalg.norm(e2)
    e2 /= n2
    e3 = np.cross(e1, e2)
    Q = np.column_stack((e1, e2, e3))
    D = Q @ np.diag([l1, l2, l3]) @ Q.T
    return -D @ (end_vel - fx)


def find_link_index(robot_id, name):
    """Return the link index whose child-link name matches *name*."""
    for i in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, i)[12].decode("utf-8") == name:
            return i
    raise ValueError(f"Link {name!r} not found")


def setup_collision_filters(robot_id):
    """Exclude the (link5, link7) pair for each arm."""
    link5s, link7s = [], []
    for i in range(p.getNumJoints(robot_id)):
        name = p.getJointInfo(robot_id, i)[12].decode("utf-8")
        if name.endswith("_link5"):
            link5s.append(i)
        elif name.endswith("_link7"):
            link7s.append(i)
    for l5 in link5s:
        for l7 in link7s:
            if abs(l5 - l7) < 5:
                p.setCollisionFilterPair(
                    robot_id, robot_id, l5, l7, enableCollision=0)


def compute_gamma_and_grad_dual(model, q_14, qd_14, threshold, device):
    """Evaluate dual-arm Gamma and, if below threshold, its gradient."""
    q_t = torch.tensor(q_14, dtype=torch.float32).unsqueeze(0)
    qd_t = torch.tensor(qd_14, dtype=torch.float32).unsqueeze(0)
    x = torch.cat([q_t, qd_t], dim=1).to(device)
    x.requires_grad_(True)
    _, gamma = model(x)
    gamma_val = gamma.item()
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

    # Hardware limits (same per arm, tiled to 14 DOFs)
    acc_max = np.tile([15, 7.5, 10, 12.5, 15, 20, 20], 2).astype(np.float32)
    qd_lim = np.tile(
        [2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], 2).astype(np.float32)
    q_min_hw = np.tile(
        [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], 2)
    q_max_hw = np.tile(
        [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], 2)

    # Impedance gains
    lambda1, lambda2, lambda3 = 5, 100, 100

    # ----- PyBullet setup -----
    p.connect(p.GUI, options="--width=1920 --height=1080")
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(
        cameraDistance=1.2, cameraYaw=90,
        cameraPitch=-20, cameraTargetPosition=[0, 0, 0.5])
    p.resetSimulation()
    p.setTimeStep(args.stepsize)
    p.setRealTimeSimulation(0)
    p.setGravity(0, 0, 0)

    # Ground plane
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane = p.loadURDF("plane.urdf", useFixedBase=True)
    p.changeDynamics(plane, -1, restitution=0.95)

    # Dual-arm robot
    urdf_path = os.path.join(
        _PROJECT_ROOT, "assets", "urdf", "panda", "panda_dual_arms.urdf")
    robot = p.loadURDF(
        urdf_path, useFixedBase=True,
        flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
    p.changeDynamics(robot, -1, linearDamping=0, angularDamping=0)

    setup_collision_filters(robot)

    # Discover movable joints
    movable_joints = []
    for i in range(p.getNumJoints(robot)):
        if p.getJointInfo(robot, i)[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            movable_joints.append(i)
    assert len(movable_joints) == N_DOF, (
        f"Expected {N_DOF} DOFs, found {len(movable_joints)}")

    # End-effector link indices
    left_ee = find_link_index(robot, "panda_left_EndEffector")
    right_ee = find_link_index(robot, "panda_right_EndEffector")

    # Disable default velocity controller (required for torque mode)
    p.setJointMotorControlArray(
        robot, movable_joints, p.VELOCITY_CONTROL, forces=[0.0] * N_DOF)

    # Initial configuration – standard home pose
    q0 = Q_HOME + Q_HOME
    for jid, angle in zip(movable_joints, q0):
        p.resetJointState(robot, jid, angle)

    # Target markers (left=blue, right=red)
    target_left = np.array(args.target_left)
    target_right = np.array(args.target_right)
    vis_l = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.03, rgbaColor=[0, 0.5, 1, 1])
    p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=vis_l,
        basePosition=target_left.tolist())
    vis_r = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.03, rgbaColor=[1, 0.3, 0, 1])
    p.createMultiBody(
        baseMass=0, baseVisualShapeIndex=vis_r,
        basePosition=target_right.tolist())

    # ----- Load dual-arm Gamma model -----
    device = torch.device("cpu")
    gamma_model = TransformerGamma(input_dim=N_DOF * 2).to(device)
    gamma_model.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=True))
    gamma_model.eval()

    # Body links for inter-arm distance check (link1..link7 per arm)
    left_body_links = list(range(1, 8))
    right_body_links = list(range(10, 17))

    # ----- Data loggers -----
    log = {k: [] for k in [
        "time", "collision_dist", "inter_arm_dist", "gamma",
        "target_dist_left", "target_dist_right", "runtime_ms",
    ]}

    time.sleep(2)
    wall_start = time.time()
    num_steps = int(args.duration / args.stepsize)
    sim_t = 0.0
    zeros_dof = [0.0] * N_DOF

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

        dist_left = np.linalg.norm(left_pos - target_left)
        dist_right = np.linalg.norm(right_pos - target_right)

        # --- Impedance force for each arm (crossing targets) ---
        fc_left = compute_impedance_force(
            left_pos, left_vel, target_left, lambda1, lambda2, lambda3)
        fc_right = compute_impedance_force(
            right_pos, right_vel, target_right, lambda1, lambda2, lambda3)

        # --- Viability stopping position ---
        qe = compute_qe(q, qd, acc_limits=DUAL_ACC_LIMITS)

        # --- Self/inter-arm collision safety ---
        Gamma, grad_gamma = compute_gamma_and_grad_dual(
            gamma_model, q, qd, args.gamma_threshold, device)

        # --- Real-time status ---
        if step_i % 100 == 0:
            freq = 1.0 / (time.perf_counter() - t0) if step_i > 0 else 0
            print(f"[sim] t={sim_t:.3f}s  Gamma={Gamma:+.4f}"
                  f"  avoidance={'ON' if grad_gamma is not None else 'off'}")
                #   f"  freq={freq:.0f}Hz")

        # --- Acceleration bounds ---
        qdd_lb, qdd_ub = compute_joint_acceleration_bounds_vec(
            q_np, qd_np, q_min_hw, q_max_hw, qd_lim, acc_max,
            dt=0.02, viability=True)

        # --- Jacobians (each 3 x 14, block-diagonal structure) ---
        dof_pos = list(q)
        J_left = np.array(p.calculateJacobian(
            robot, left_ee, [0, 0, 0], dof_pos, zeros_dof, zeros_dof)[0])
        J_right = np.array(p.calculateJacobian(
            robot, right_ee, [0, 0, 0], dof_pos, zeros_dof, zeros_dof)[0])
        J_stack = np.vstack([J_left, J_right])
        fc_stack = np.concatenate([fc_left, fc_right])
        JT_pinv = np.linalg.pinv(J_stack.T)

        # --- Dynamics ---
        M = np.array(p.calculateMassMatrix(robot, dof_pos))
        tau_id = np.array(p.calculateInverseDynamics(
            robot, dof_pos, list(qd_np), zeros_dof))
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
                tau = list(p.calculateInverseDynamics(
                    robot, dof_pos, list(qd_np), qdd_cmd.tolist()))
                p.setJointMotorControlArray(
                    robot, movable_joints, p.TORQUE_CONTROL, forces=tau)
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
        log["target_dist_left"].append(dist_left)
        log["target_dist_right"].append(dist_right)
        log["runtime_ms"].append((t1 - t0) * 1e3)

        if collision:
            print(f"[sim] t={sim_t:.3f}s  COLLISION  dist={collision_dist:.5f}"
                  f"  inter_arm={inter_dist:.5f}  Gamma={Gamma:+.4f}")
            break

        time.sleep(args.stepsize)

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
    csv_path = os.path.join(args.output_dir, f"dual_fixed_{int(time.time())}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
