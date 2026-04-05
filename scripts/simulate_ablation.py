#!/usr/bin/env python3
"""Single-arm self-collision avoidance test (no external obstacles).

Drives the Panda end-effector toward a target point using impedance control
with only the TransformerGamma self-collision avoidance active.

Usage
-----
    python scripts/simulate_ablation.py
    python scripts/simulate_ablation.py --model-path transformer_gamma_ablation.pt
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

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, os.pardir)
sys.path.insert(0, os.path.abspath(_PROJECT_ROOT))

import torch

from vpptc.model import TransformerGamma
from vpptc.robot import Panda
from vpptc.safety import compute_joint_acceleration_bounds_vec
from vpptc.utils import compute_qe


def load_gamma_model(model_path):
    """Load TransformerGamma model explicitly."""
    device = torch.device("cpu")
    model = TransformerGamma().to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, device


def compute_gamma_and_grad_local(model, q, qd, threshold, device):
    """Compute Gamma and gradient using the explicitly loaded model."""
    q_t = torch.tensor(q, dtype=torch.float32).unsqueeze(0)
    qd_t = torch.tensor(qd, dtype=torch.float32).unsqueeze(0)
    x = torch.cat([q_t, qd_t], dim=1).to(device)
    x.requires_grad_(True)
    _, gamma = model(x)
    gamma_val = gamma.item()
    if gamma_val < threshold:
        gamma.backward()
        return gamma_val, x.grad.squeeze(0).cpu().numpy()
    return gamma_val, None


# ======================================================================
# Argument parsing
# ======================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Single-arm self-collision test (no external obstacles)",
    )
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Simulation duration in seconds (default: 10.0)")
    parser.add_argument("--stepsize", type=float, default=2e-3,
                        help="Simulation time step (default: 2e-3)")
    parser.add_argument("--seed", type=int, default=28,
                        help="Random seed (default: 28)")
    parser.add_argument("--target-pos", type=float, nargs=3,
                        default=[0.0, 0.0, 0.3],
                        help="End-effector target position [x y z]")
    parser.add_argument("--gamma-threshold", type=float, default=2.5,
                        help="Self-collision Gamma threshold (default: 2.5)")
    parser.add_argument("--alpha", type=float, default=1e-2,
                        help="Regularisation weight in the QP (default: 1e-2)")
    parser.add_argument("--model-path", type=str,
                        default=os.path.join(
                            _PROJECT_ROOT, "assets", "models",
                            "transformer_gamma.pt"),
                        help="Path to TransformerGamma model weights")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(_PROJECT_ROOT, "output"),
                        help="Directory for result CSV files")
    return parser.parse_args()


# ======================================================================
# Main
# ======================================================================

def main():
    args = get_args()
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ----- Load Gamma model -----
    print(f"Loading model: {args.model_path}")
    gamma_model, device = load_gamma_model(args.model_path)

    # Hardware limits for Franka Panda
    acc_max = np.array([15, 7.5, 10, 12.5, 15, 20, 20], dtype=np.float32)
    qd_lim = np.array([2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61], dtype=np.float32)
    q_min_hw = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    q_max_hw = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    # Impedance gains
    lambda1, lambda2, lambda3 = 5, 100, 100

    # ----- Initialise robot -----
    robot = Panda(stepsize=args.stepsize)
    robot.setControlMode("torque")

    # Target marker (green)
    target_pos = np.array(args.target_pos)
    target_vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 1]
    )
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=target_vis,
                      basePosition=target_pos.tolist())

    # ----- Data loggers -----
    log = {k: [] for k in [
        "time", "self_collision_dist", "gamma",
        "target_dist", "runtime_ms",
    ]}

    time.sleep(2)
    wall_start = time.time()
    num_steps = int(args.duration / args.stepsize)

    for step_i in range(num_steps):
        t0 = time.perf_counter()

        # --- End-effector impedance force ---
        end_pos = np.array(robot.solveForwardKinematics()[0])
        fx = -50.0 * (end_pos - target_pos)
        e1 = fx / np.linalg.norm(fx)
        e2 = np.array([1, 0, 0]) - np.dot([1, 0, 0], e1) * e1
        e2 /= np.linalg.norm(e2)
        e3 = np.cross(e1, e2)
        Q = np.column_stack((e1, e2, e3))
        D = Q @ np.diag([lambda1, lambda2, lambda3]) @ Q.T
        xdot = np.array(robot.getEndVelocity())
        fc = -D @ (xdot - fx)

        dist_to_target = np.linalg.norm(end_pos - target_pos)

        # --- Joint state & viability stopping position ---
        q, qd = robot.getJointStates()
        qe = compute_qe(q, qd)

        # --- Self-collision safety ---
        Gamma, grad_gamma = compute_gamma_and_grad_local(
            gamma_model, q, qd, args.gamma_threshold, device)

        # --- Print status ---
        if step_i % 100 == 0:
            sca_status = "ON" if grad_gamma is not None else "off"
            print(f"[sim] t={robot.t:.3f}s  Gamma={Gamma:+.4f}  SCA={sca_status}"
                  f"  target_dist={dist_to_target:.4f}")

        # --- Acceleration bounds ---
        qdd_lb, qdd_ub = compute_joint_acceleration_bounds_vec(
            q, qd, q_min_hw, q_max_hw, qd_lim, acc_max, dt=0.02, viability=True)

        # --- Ensure feasibility ---
        for idx in range(7):
            if qdd_lb[idx] > qdd_ub[idx]:
                qdd_lb[idx] = qdd_ub[idx] - 1e-4

        M = np.array(robot.getMassMatrix(q))
        tau_id = np.array(robot.solveInverseDynamics(q, qd, [0] * 7))
        M_inv = np.linalg.inv(M)

        u = cp.Variable(7)
        J = np.array(robot.getJacobian())
        JT_pinv = np.linalg.pinv(J.T)

        objective = (cp.sum_squares(JT_pinv @ u - fc)
                     + args.alpha * cp.sum_squares(u))
        constraints = [
            M_inv @ u >= qdd_lb + M_inv @ tau_id,
            M_inv @ u <= qdd_ub + M_inv @ tau_id,
        ]

        soft = False
        if grad_gamma is not None:
            dt = 0.02
            gq = grad_gamma[:7]
            gqd = grad_gamma[7:]
            g_eff = 0.5 * gq * dt ** 2 + gqd * dt
            c_const = gq.dot(qd) * dt
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
                tau = robot.solveInverseDynamics(q, qd, qdd_cmd.tolist())
                robot.setTargetTorques(tau)
                robot.step()
        else:
            prob = cp.Problem(cp.Minimize(objective), constraints)
            prob.solve(solver=cp.OSQP)

        if not soft:
            tau_cmd = np.array(u.value)
            robot.setTargetTorques(tau_cmd.tolist())
            robot.step()

        t1 = time.perf_counter()

        # --- Self-collision distance check ---
        sc_dist = math.inf
        collision = False
        for l1 in range(7):
            for l2 in range(7):
                if abs(l1 - l2) > 1 and {l1, l2} != {4, 6}:
                    pts = robot.getClosestPoints(l1, l2)
                    dmin = min(pt[8] for pt in pts)
                    sc_dist = min(sc_dist, dmin)
                    if dmin < 0:
                        collision = True

        # --- Log ---
        log["time"].append(robot.t)
        log["self_collision_dist"].append(sc_dist)
        log["gamma"].append(Gamma)
        log["target_dist"].append(dist_to_target)
        log["runtime_ms"].append((t1 - t0) * 1e3)

        if collision:
            print(f"[sim] t={robot.t:.3f}s  COLLISION  dist={sc_dist:.5f}  Gamma={Gamma:+.4f}")

        time.sleep(args.stepsize)

    # --- Summary ---
    elapsed = time.time() - wall_start
    runtimes = np.array(log["runtime_ms"][1:])
    print(f"\n{'='*50}")
    print(f"Simulation finished in {elapsed:.2f}s wall-clock")
    print(f"  Steps         : {len(log['time'])}")
    if len(runtimes) > 0:
        print(f"  Mean step     : {runtimes.mean():.3f} ms")
        print(f"  Median step   : {np.median(runtimes):.3f} ms")
        print(f"  95th pct      : {np.percentile(runtimes, 95):.3f} ms")
        print(f"  Max step      : {runtimes.max():.3f} ms")

    # --- Save ---
    df = pd.DataFrame(log)
    csv_path = os.path.join(args.output_dir, f"ablation_run_{int(time.time())}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
