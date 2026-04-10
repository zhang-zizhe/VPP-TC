#!/usr/bin/env python3
"""Single-arm limit-cycle trajectory test.

Drives one Panda arm along a Hopf-oscillator limit cycle using
Jacobian-transpose torque control.  No collision avoidance – purely
for verifying that the DS + controller can track the circle.

Usage
-----
    python scripts/traj.py
    python scripts/traj.py --plane xz --radius 0.15 --omega 2.0
    python scripts/traj.py --center 0.4 0.0 0.5 --radius 0.12 --duration 20
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import pybullet as p

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, os.pardir)
sys.path.insert(0, os.path.abspath(_PROJECT_ROOT))

from vpptc.robot import Panda

Q_HOME = [0, -0.785, 0, -2.356, 0, 1.571, 0]
_PLANE_AXES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}


def get_args():
    ap = argparse.ArgumentParser(description="Single-arm limit-cycle test")
    ap.add_argument("--duration", type=float, default=15.0)
    ap.add_argument("--stepsize", type=float, default=1e-3)
    ap.add_argument("--center", type=float, nargs=3, default=[0.4, 0.0, 0.5])
    ap.add_argument("--radius", type=float, default=0.12)
    ap.add_argument("--omega", type=float, default=3.0)
    ap.add_argument("--plane", type=str, default="xz",
                    choices=["xy", "xz", "yz"])
    ap.add_argument("--k-d", type=float, default=150.0,
                    help="Velocity-tracking damping gain")
    ap.add_argument("--k-pos", type=float, default=80.0,
                    help="Radial position-correction gain")
    ap.add_argument("--alpha", type=float, default=20.0,
                    help="Hopf radial convergence gain")
    ap.add_argument("--k-perp", type=float, default=20.0,
                    help="Out-of-plane convergence gain")
    return ap.parse_args()


MAX_DS_SPEED = 0.6          # m/s  – cap the DS desired velocity


def limit_cycle_ds(pos, center, radius, omega, plane, alpha, k_perp):
    """Hopf-oscillator desired velocity in an arbitrary 2-D plane."""
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


def compute_ds_force(pos, vel, center, radius, omega, plane,
                     k_d, k_pos, alpha, k_perp):
    """Velocity-tracking + position-correction Cartesian force."""
    i1, i2, i3 = _PLANE_AXES[plane]
    v_des = limit_cycle_ds(pos, center, radius, omega, plane, alpha, k_perp)

    f_pos = np.zeros(3)
    # In-plane: radial spring toward the circle
    d = np.array([pos[i1] - center[i1], pos[i2] - center[i2]])
    rho = np.linalg.norm(d)
    if rho > 1e-8:
        direction = d / rho
        f_pos[i1] = -k_pos * (rho - radius) * direction[0]
        f_pos[i2] = -k_pos * (rho - radius) * direction[1]
    # Out-of-plane: spring back to the circle's plane
    f_pos[i3] = -k_pos * (pos[i3] - center[i3])

    return k_d * (v_des - vel) + f_pos


def draw_circle(center, radius, plane, color, n_seg=60):
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


def main():
    args = get_args()
    center = np.array(args.center)

    robot = Panda(stepsize=args.stepsize, realtime=0)

    # Reset to home pose
    for j in range(7):
        p.resetJointState(robot.robot, j, Q_HOME[j])
    robot.resetController()

    p.resetDebugVisualizerCamera(
        cameraDistance=1.0, cameraYaw=60,
        cameraPitch=-25, cameraTargetPosition=[0.3, 0, 0.4])

    # Draw limit cycle + centre marker
    draw_circle(center, args.radius, args.plane, color=[0.2, 0.8, 0.3])
    vis = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.012, rgbaColor=[0.2, 0.8, 0.3, 0.6])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                      basePosition=center.tolist())

    num_steps = int(args.duration / args.stepsize)
    prev_ee = None
    trail_color = [1.0, 0.4, 0.0]

    time.sleep(1)
    print(f"[traj] plane={args.plane}  center={args.center}  "
          f"r={args.radius}  omega={args.omega}")

    for step_i in range(num_steps):
        q, qd = robot.getJointStates()

        ee_state = p.getLinkState(robot.robot, 7, computeLinkVelocity=True)
        ee_pos = np.array(ee_state[0])
        ee_vel = np.array(ee_state[6])

        # DS force
        fc = compute_ds_force(
            ee_pos, ee_vel, center, args.radius, args.omega, args.plane,
            args.k_d, args.k_pos, args.alpha, args.k_perp)

        # Jacobian-transpose torque: tau = J^T * f
        J = np.array(robot.getJacobian())          # (3, 7)
        tau_task = J.T @ fc                         # (7,)

        # Gravity/Coriolis compensation
        tau_id = np.array(p.calculateInverseDynamics(
            robot.robot, q, qd, [0.0] * 7))
        # Joint-space damping for stability
        tau_damp = -8.0 * np.array(qd)

        tau = tau_task + tau_id + tau_damp
        # Panda torque limits: joints 1-4 ≈ 87 Nm, joints 5-7 ≈ 12 Nm
        tau_max = np.array([87, 87, 87, 87, 12, 12, 12], dtype=float)
        tau = np.clip(tau, -tau_max, tau_max)
        robot.setTargetTorques(tau.tolist())
        robot.step()

        # EE trail
        if step_i % 20 == 0:
            if prev_ee is not None:
                p.addUserDebugLine(
                    prev_ee.tolist(), ee_pos.tolist(),
                    trail_color, lineWidth=1.5, lifeTime=8)
            prev_ee = ee_pos.copy()

        # Status
        if step_i % 1000 == 0:
            i1, i2, _ = _PLANE_AXES[args.plane]
            d = np.array([ee_pos[i1] - center[i1], ee_pos[i2] - center[i2]])
            rho = np.linalg.norm(d)
            err = abs(rho - args.radius)
            print(f"  t={step_i * args.stepsize:6.2f}s  "
                  f"radial_err={err:.4f}m  rho={rho:.4f}")

        time.sleep(args.stepsize)

    print("[traj] Done.")


if __name__ == "__main__":
    main()
