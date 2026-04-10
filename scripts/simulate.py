#!/usr/bin/env python3
"""VPP-TC: main simulation script.

Runs a Franka Panda robot in PyBullet with viability-preserving torque control.
Both self-collision avoidance (via the TransformerGamma model) and external
collision avoidance (via RDF signed distance fields) are active.

Usage
-----
    python scripts/simulate.py
    python scripts/simulate.py --duration 10 --stepsize 2e-3 --seed 42
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

# Ensure the project root is on ``sys.path`` so that ``vpptc`` and
# ``third_party`` are importable regardless of where we run from.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, os.pardir)
sys.path.insert(0, os.path.abspath(_PROJECT_ROOT))

from vpptc.robot import Panda
from vpptc.safety import (
    compute_gamma_and_grad,
    compute_joint_acceleration_bounds_vec,
)
from vpptc.sdf import query_sdf_batch
from vpptc.utils import compute_min_center_distance, compute_qe


# ======================================================================
# Argument parsing
# ======================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="VPP-TC simulation with self- and external collision avoidance",
    )
    # Simulation
    parser.add_argument("--duration", type=float, default=6.0,
                        help="Simulation duration in seconds (default: 6.0)")
    parser.add_argument("--stepsize", type=float, default=2e-3,
                        help="Simulation time step (default: 2e-3)")
    parser.add_argument("--seed", type=int, default=28,
                        help="Random seed (default: 28)")

    # Obstacle
    parser.add_argument("--obstacle-pos", type=float, nargs=3,
                        default=[0.0, -0.4, 0.5],
                        help="Obstacle 1 centre [x y z]")
    parser.add_argument("--obstacle-radius", type=float, default=0.05,
                        help="Obstacle sphere radius (default: 0.05)")

    # Target
    parser.add_argument("--target-pos", type=float, nargs=3,
                        default=[0.0, -0.6, 0.3],
                        help="End-effector target position [x y z]")

    # Safety
    parser.add_argument("--gamma-threshold", type=float, default=2.5,
                        help="Self-collision Gamma threshold (default: 2.5)")
    parser.add_argument("--sdf-react-dist", type=float, default=0.1,
                        help="SDF distance below which reactive evasion activates")

    # Controller
    parser.add_argument("--alpha", type=float, default=1e-2,
                        help="Regularisation weight in the QP (default: 1e-2)")

    # Output
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(_PROJECT_ROOT, "output"),
                        help="Directory for result CSV files")
    return parser.parse_args()


# ======================================================================
# Helper: create a sphere obstacle in PyBullet
# ======================================================================

def create_sphere(centre, radius, rgba=(1, 0, 0, 1)):
    """Create a static sphere in PyBullet and return its body id."""
    col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
    return p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=centre,
    )


# ======================================================================
# Main
# ======================================================================

def main():
    args = get_args()
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

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

    # ----- Create obstacles -----
    x_obs = list(args.obstacle_pos)
    obstacle_1 = create_sphere(x_obs, args.obstacle_radius, rgba=(1, 0, 0, 1))
    obstacle_2 = create_sphere(
        [x_obs[0] + 0.5, x_obs[1] + 0.1, x_obs[2]],
        args.obstacle_radius,
        rgba=(1, 0, 0, 1),
    )

    # Target marker (green)
    target_pos = np.array(args.target_pos)
    target_vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.02, rgbaColor=[0, 1, 0, 1]
    )
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=target_vis,
                      basePosition=target_pos.tolist())

    # ----- Data loggers -----
    log = {k: [] for k in [
        "time", "self_collision_dist", "gamma", "real_dist",
        "target_dist", "pred_dist", "pred_dist_viability", "runtime_ms",
    ]}

    time.sleep(2)
    wall_start = time.time()
    num_steps = int(args.duration / args.stepsize)

    for step_i in range(num_steps):
        t0 = time.perf_counter()

        if step_i % int(1.0 / args.stepsize) == 0:
            print(f"[sim] t = {robot.t:.3f} s")

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

        # --- Move obstacle sinusoidally ---
        new_z = math.sin(step_i / 180.0 * math.pi) * 0.1
        new_pos = [x_obs[0], x_obs[1], x_obs[2] + new_z]
        p.resetBasePositionAndOrientation(obstacle_1, new_pos, [0, 0, 0, 1])

        # --- Joint state & viability stopping position ---
        q, qd = robot.getJointStates()
        qe = compute_qe(q, qd)

        # --- Batch SDF query (2 configs x 2 points) ---
        x0 = np.array(x_obs, dtype=np.float32).reshape(1, 3)
        x_query = np.array(new_pos, dtype=np.float32).reshape(1, 3)
        pose = np.eye(4, dtype=np.float32)

        theta_np = np.stack([q, qe], axis=0).astype(np.float32)
        points_np = np.stack([
            x_query.squeeze(),
            (x0 + np.array([0.5, 0.1, 0.0], dtype=np.float32)).squeeze(),
        ], axis=0).astype(np.float32)
        pose_np = np.broadcast_to(pose, (2, 4, 4)).astype(np.float32)

        dsts, grad_qs = query_sdf_batch(points_np, pose_np, theta_np)

        dst, grad = dsts[0, 0], grad_qs[0, 0]
        dst2, grad2 = dsts[1, 0], grad_qs[1, 0]
        dst3, grad3 = dsts[0, 1], grad_qs[0, 1]
        dst4, grad4 = dsts[1, 1], grad_qs[1, 1]

        # --- Real PyBullet distance (ground-truth check) ---
        real_dist1 = compute_min_center_distance(
            robot.robot, obstacle_1, args.obstacle_radius, 2.0)
        real_dist2 = compute_min_center_distance(
            robot.robot, obstacle_2, args.obstacle_radius, 2.0)

        # --- Self-collision safety ---
        Gamma, grad_gamma = compute_gamma_and_grad(
            q, qd, threshold=args.gamma_threshold)

        # --- Acceleration bounds ---
        qdd_lb, qdd_ub = compute_joint_acceleration_bounds_vec(
            q, qd, q_min_hw, q_max_hw, qd_lim, acc_max, dt=0.02, viability=True)

        # --- Select gradient for the closest obstacle configuration ---
        all_dsts = [dst, dst2, dst3, dst4]
        all_grads = [grad, grad2, grad3, grad4]
        min_idx = int(np.argmin(all_dsts))
        sel_grad = all_grads[min_idx]

        # --- Reactive evasion or QP controller ---
        if min(all_dsts) < args.sdf_react_dist:
            # Emergency: directly steer away from obstacle
            dt = 0.02
            g_eff = 0.5 * sel_grad * dt ** 2
            qdd_cmd = np.where(g_eff > 0, qdd_ub, qdd_lb)
            tau = robot.solveInverseDynamics(q, qd, qdd_cmd.tolist())
            robot.setTargetTorques(tau)
            robot.step()
        else:
            # Ensure feasibility
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
        log["real_dist"].append(min(real_dist1, real_dist2))
        log["target_dist"].append(dist_to_target)
        log["pred_dist"].append(min(dst, dst3))
        log["pred_dist_viability"].append(min(dst2, dst4))
        log["runtime_ms"].append((t1 - t0) * 1e3)

        if collision:
            print(f"[sim] t={robot.t:.3f}s  COLLISION  dist={sc_dist:.5f}")
            break

        time.sleep(args.stepsize)

    # --- Summary ---
    elapsed = time.time() - wall_start
    runtimes = np.array(log["runtime_ms"][1:])
    print(f"\n{'='*50}")
    print(f"Simulation finished in {elapsed:.2f}s wall-clock")
    print(f"  Steps         : {len(log['time'])}")
    print(f"  Mean step     : {runtimes.mean():.3f} ms")
    print(f"  Median step   : {np.median(runtimes):.3f} ms")
    print(f"  95th pct      : {np.percentile(runtimes, 95):.3f} ms")
    print(f"  Max step      : {runtimes.max():.3f} ms")

    # --- Save ---
    df = pd.DataFrame(log)
    csv_path = os.path.join(args.output_dir, f"run_{int(time.time())}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
