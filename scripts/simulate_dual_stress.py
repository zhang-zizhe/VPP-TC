#!/usr/bin/env python3
"""Dual-arm Panda simulation with VPP-TC inter-arm collision avoidance.

Each arm's end-effector follows a stable limit-cycle dynamical system
(Hopf oscillator).  The two limit cycles overlap in the shared workspace,
forcing the arms to repeatedly cross paths and stress-test the
TransformerGamma collision avoidance.

Usage
-----
    python scripts/simulate_dual.py
    python scripts/simulate_dual.py --duration 20 --lc-radius 0.15 --lc-omega 3.0
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
        description="Dual-arm VPP-TC simulation (inter-arm collision avoidance)",
    )
    parser.add_argument("--duration", type=float, default=20,
                        help="Simulation duration in seconds (default: 6.0)")
    parser.add_argument("--stepsize", type=float, default=2e-3,
                        help="Simulation time step (default: 2e-3)")
    parser.add_argument("--seed", type=int, default=28,
                        help="Random seed (default: 28)")
    parser.add_argument("--lc-center-left", type=float, nargs=3,
                        default=[0.3, 0.08, 0.45],
                        help="Left-arm limit cycle center [x y z]")
    parser.add_argument("--lc-center-right", type=float, nargs=3,
                        default=[0.3, -0.08, 0.45],
                        help="Right-arm limit cycle center [x y z]")
    parser.add_argument("--lc-radius", type=float, default=0.2,
                        help="Limit cycle radius in metres (default: 0.15)")
    parser.add_argument("--lc-omega", type=float, default=2.5,
                        help="Limit cycle angular speed (default: 3.0 rad/s)")
    parser.add_argument("--lc-plane-left", type=str, default="xz",
                        choices=["xy", "xz", "yz"],
                        help="Plane for the left-arm limit cycle (default: xz)")
    parser.add_argument("--lc-plane-right", type=str, default="xy",
                        choices=["xy", "xz", "yz"],
                        help="Plane for the right-arm limit cycle (default: xy)")
    parser.add_argument("--gamma-threshold", type=float, default=4,
                        help="Gamma threshold for collision avoidance")
    parser.add_argument("--alpha", type=float, default=1e-3,
                        help="Regularisation weight in the QP")
    parser.add_argument("--model-path", type=str,
                        default=os.path.join(
                            _PROJECT_ROOT, "assets", "models",
                            "transformer_gamma_dual_d0.6.pt"),
                        help="Path to the dual-arm TransformerGamma weights")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(_PROJECT_ROOT, "output"),
                        help="Directory for result CSV files")
    parser.add_argument("--lc-shrink-rate", type=float, default=0.0,
                        help="Radius shrink rate in m/s (0=static)")
    parser.add_argument("--lc-radius-min", type=float, default=0.05,
                        help="Minimum radius when shrinking (default: 0.05)")
    parser.add_argument("--lc-rotate-speed-left", type=float, default=0.0,
                        help="Left plane rotation speed in rad/s (0=no rotation)")
    parser.add_argument("--lc-rotate-speed-right", type=float, default=0.0,
                        help="Right plane rotation speed in rad/s (0=no rotation)")
    parser.add_argument("--lc-rotate-axis-left", type=float, nargs=3,
                        default=[1.0, 0.0, 0.0],
                        help="Left rotation axis [x y z] (default: [1 0 0])")
    parser.add_argument("--lc-rotate-axis-right", type=float, nargs=3,
                        default=[0.5, 0.0, 0.0],
                        help="Right rotation axis [x y z] (default: [1 0 0])")
    parser.add_argument("--lc-kd", type=float, default=200,
                        help="DS velocity tracking gain (default: 150)")
    parser.add_argument("--lc-kpos", type=float, default=80.0,
                        help="Position correction gain toward the cycle (default: 80)")
    parser.add_argument("--lc-alpha", type=float, default=20.0,
                        help="Hopf radial convergence rate (default: 20)")
    parser.add_argument("--lc-kperp", type=float, default=20.0,
                        help="Out-of-plane restoring stiffness (default: 20)")
    parser.add_argument("--lc-eccentricity", type=float, default=0.0,
                        help="Max ellipse deformation ratio (0=circle, 0.5=moderate)")
    parser.add_argument("--lc-eccen-freq", type=float, default=0.1,
                        help="Eccentricity oscillation frequency in Hz (default: 0.5)")
    parser.add_argument("--lc-center-speed", type=float, default=0.01,
                        help="Speed at which LC centres approach each other in m/s (0=static)")
    return parser.parse_args()


# ======================================================================
# Helpers
# ======================================================================

_PLANE_AXES = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}
MAX_DS_SPEED = 0.5


def limit_cycle_ds(pos, center, radius, omega, plane="xy",
                   alpha=20.0, k_perp=20.0):
    """Hopf-oscillator stable limit cycle in an arbitrary 2-D plane.

    The two in-plane axes follow the Hopf oscillator; the out-of-plane
    axis has independent linear convergence to ``center``.
    Velocity is capped at MAX_DS_SPEED to prevent overshooting.
    """
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
                     k_d=150.0, k_pos=80.0, alpha=20.0, k_perp=20.0):
    """Velocity-tracking + position-correction force for the limit cycle DS.

    ``k_d`` tracks the DS velocity; ``k_pos`` adds a radial spring that
    pulls the EE back toward the circle when it drifts off (in-plane
    and out-of-plane).
    """
    i1, i2, i3 = _PLANE_AXES[plane]
    v_des = limit_cycle_ds(end_pos, center, radius, omega, plane,
                           alpha=alpha, k_perp=k_perp)

    f_pos = np.zeros(3)
    # In-plane: radial spring toward the circle
    d = np.array([end_pos[i1] - center[i1], end_pos[i2] - center[i2]])
    rho = np.linalg.norm(d)
    if rho > 1e-8:
        direction = d / rho
        f_pos[i1] = -k_pos * (rho - radius) * direction[0]
        f_pos[i2] = -k_pos * (rho - radius) * direction[1]
    # Out-of-plane: spring back to the circle's plane
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


def plane_vectors(plane):
    """Return orthonormal (u, v) vectors for the named plane."""
    i1, i2, _ = _PLANE_AXES[plane]
    u = np.zeros(3); u[i1] = 1.0
    v = np.zeros(3); v[i2] = 1.0
    return u, v


def rotate_vector(vec, axis, angle):
    """Rodrigues' rotation formula."""
    k = axis / np.linalg.norm(axis)
    c, s = math.cos(angle), math.sin(angle)
    return vec * c + np.cross(k, vec) * s + k * np.dot(k, vec) * (1 - c)


def limit_cycle_ds_uv(pos, center, ru, rv, omega, u, v,
                       alpha=20.0, k_perp=20.0):
    """Hopf-oscillator limit cycle (ellipse) in an arbitrary (u, v) plane.

    *ru* / *rv* are semi-axes along u / v.  When ru == rv it reduces to
    the standard circular Hopf oscillator.
    """
    d = pos - center
    d1, d2 = np.dot(d, u), np.dot(d, v)
    n = np.cross(u, v)
    d3 = np.dot(d, n)
    rho_sq_n = (d1 / ru) ** 2 + (d2 / rv) ** 2
    radial = alpha * ru * rv * (1 - rho_sq_n)
    vel = ((radial * d1 - omega * (ru / rv) * d2) * u
           + (radial * d2 + omega * (rv / ru) * d1) * v
           + (-k_perp * d3) * n)
    speed = np.linalg.norm(vel)
    if speed > MAX_DS_SPEED:
        vel *= MAX_DS_SPEED / speed
    return vel


def compute_ds_force_uv(end_pos, end_vel, center, ru, rv, omega, u, v,
                         k_d=150.0, k_pos=80.0, alpha=20.0, k_perp=20.0):
    """Velocity-tracking + position-correction force for (u, v) ellipse LC."""
    v_des = limit_cycle_ds_uv(end_pos, center, ru, rv, omega, u, v,
                               alpha=alpha, k_perp=k_perp)
    d = end_pos - center
    d1, d2 = np.dot(d, u), np.dot(d, v)
    n = np.cross(u, v)
    d3 = np.dot(d, n)
    rho_n = math.sqrt((d1 / ru) ** 2 + (d2 / rv) ** 2)
    f_pos = np.zeros(3)
    if rho_n > 1e-8:
        f_pos += -k_pos * (1 - 1 / rho_n) * (d1 * u + d2 * v)
    f_pos += -k_pos * d3 * n
    return k_d * (v_des - end_vel) + f_pos


def radial_error_uv(pos, center, ru, rv, u, v):
    """Approx. distance from *pos* to the ellipse in the (u, v) plane."""
    d = pos - center
    d1, d2 = np.dot(d, u), np.dot(d, v)
    rho = math.sqrt(d1 ** 2 + d2 ** 2)
    rho_n = math.sqrt((d1 / ru) ** 2 + (d2 / rv) ** 2)
    if rho_n < 1e-8:
        return max(ru, rv)
    return rho * abs(1 - 1 / rho_n)


def draw_circle_uv(center, ru, rv, color, u, v, n_seg=30, line_ids=None):
    """Visualise a limit cycle ellipse in an arbitrary (u, v) plane.

    Pass *line_ids* from a previous call to update in-place (smooth).
    Returns the list of debug-line IDs for the next call.
    """
    ids = []
    for i in range(n_seg):
        a1 = 2 * math.pi * i / n_seg
        a2 = 2 * math.pi * (i + 1) / n_seg
        pt1 = center + ru * math.cos(a1) * u + rv * math.sin(a1) * v
        pt2 = center + ru * math.cos(a2) * u + rv * math.sin(a2) * v
        kw = {}
        if line_ids is not None and i < len(line_ids):
            kw["replaceItemUniqueId"] = line_ids[i]
        ids.append(p.addUserDebugLine(
            pt1.tolist(), pt2.tolist(), color, lineWidth=2, lifeTime=0, **kw))
    return ids


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
        return gamma_val, x.grad.squeeze(0).cpu().numpy()  # (28,)
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

    # Limit cycle DS parameters (separate centres, perpendicular planes)
    lc_center_L = np.array(args.lc_center_left)
    lc_center_R = np.array(args.lc_center_right)
    lc_radius = args.lc_radius
    lc_omega = args.lc_omega
    plane_L = args.lc_plane_left
    plane_R = args.lc_plane_right

    # Dynamic limit cycle setup
    u0_L, v0_L = plane_vectors(plane_L)
    u0_R, v0_R = plane_vectors(plane_R)
    rot_axis_L = np.array(args.lc_rotate_axis_left, dtype=float)
    rot_axis_L = rot_axis_L / np.linalg.norm(rot_axis_L)
    rot_axis_R = np.array(args.lc_rotate_axis_right, dtype=float)
    rot_axis_R = rot_axis_R / np.linalg.norm(rot_axis_R)
    dynamic_lc = (args.lc_shrink_rate > 0
                  or args.lc_rotate_speed_left > 0
                  or args.lc_rotate_speed_right > 0
                  or args.lc_eccentricity > 0
                  or args.lc_center_speed > 0)

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

    # Draw limit cycle circles (static only; dynamic redraws in loop)
    if not dynamic_lc:
        draw_circle(lc_center_L, lc_radius, color=[0, 0.5, 1], plane=plane_L)
        draw_circle(lc_center_R, lc_radius, color=[1, 0.3, 0], plane=plane_R)
    # Small spheres at centres (keep IDs for dynamic update)
    vis_l = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.015, rgbaColor=[0, 0.5, 1, 0.6])
    center_body_L = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_l,
                                      basePosition=lc_center_L.tolist())
    vis_r = p.createVisualShape(
        p.GEOM_SPHERE, radius=0.015, rgbaColor=[1, 0.3, 0, 0.6])
    center_body_R = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis_r,
                                      basePosition=lc_center_R.tolist())

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
        "lc_dist_left", "lc_dist_right", "radius", "runtime_ms",
    ]}

    time.sleep(2)
    wall_start = time.time()
    num_steps = int(args.duration / args.stepsize)
    sim_t = 0.0
    zeros_dof = [0.0] * N_DOF
    circle_ids_L = None
    circle_ids_R = None

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

        # --- Dynamic limit cycle: shrink + rotate + eccentricity + approach ---
        if dynamic_lc:
            # Move centres toward each other along y-axis
            if args.lc_center_speed > 0:
                mid_y = (args.lc_center_left[1] + args.lc_center_right[1]) / 2
                drift = args.lc_center_speed * sim_t
                half_dist = abs(args.lc_center_left[1] - mid_y)
                offset = max(half_dist - drift, 0.0)
                lc_center_L[1] = mid_y + offset
                lc_center_R[1] = mid_y - offset
                p.resetBasePositionAndOrientation(
                    center_body_L, lc_center_L.tolist(), [0, 0, 0, 1])
                p.resetBasePositionAndOrientation(
                    center_body_R, lc_center_R.tolist(), [0, 0, 0, 1])

            r_now = max(lc_radius - args.lc_shrink_rate * sim_t,
                        args.lc_radius_min) if args.lc_shrink_rate > 0 else lc_radius
            if args.lc_eccentricity > 0:
                ecc = args.lc_eccentricity * math.sin(
                    2 * math.pi * args.lc_eccen_freq * sim_t)
                ru_now = r_now * (1 + ecc)
                rv_now = r_now * (1 - ecc)
            else:
                ru_now = rv_now = r_now
            if args.lc_rotate_speed_left > 0:
                ang_L = args.lc_rotate_speed_left * sim_t
                u_L = rotate_vector(u0_L, rot_axis_L, ang_L)
                v_L = rotate_vector(v0_L, rot_axis_L, ang_L)
            else:
                u_L, v_L = u0_L, v0_L
            if args.lc_rotate_speed_right > 0:
                ang_R = args.lc_rotate_speed_right * sim_t
                u_R = rotate_vector(u0_R, rot_axis_R, ang_R)
                v_R = rotate_vector(v0_R, rot_axis_R, ang_R)
            else:
                u_R, v_R = u0_R, v0_R
            circle_ids_L = draw_circle_uv(
                lc_center_L, ru_now, rv_now, [0, 0.5, 1], u_L, v_L,
                line_ids=circle_ids_L)
            circle_ids_R = draw_circle_uv(
                lc_center_R, ru_now, rv_now, [1, 0.3, 0], u_R, v_R,
                line_ids=circle_ids_R)
            dr_L = radial_error_uv(left_pos, lc_center_L, ru_now, rv_now, u_L, v_L)
            dr_R = radial_error_uv(right_pos, lc_center_R, ru_now, rv_now, u_R, v_R)
            fc_left = compute_ds_force_uv(
                left_pos, left_vel, lc_center_L, ru_now, rv_now, lc_omega, u_L, v_L,
                k_d=args.lc_kd, k_pos=args.lc_kpos,
                alpha=args.lc_alpha, k_perp=args.lc_kperp)
            fc_right = compute_ds_force_uv(
                right_pos, right_vel, lc_center_R, ru_now, rv_now, -lc_omega, u_R, v_R,
                k_d=args.lc_kd, k_pos=args.lc_kpos,
                alpha=args.lc_alpha, k_perp=args.lc_kperp)
        else:
            r_now = lc_radius
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
        J_stack = np.vstack([J_left, J_right])          # (6, 14)
        fc_stack = np.concatenate([fc_left, fc_right])   # (6,)
        JT_pinv = np.linalg.pinv(J_stack.T)              # (6, 14)

        # --- Dynamics ---
        M = np.array(p.calculateMassMatrix(robot, dof_pos))        # (14, 14)
        tau_id = np.array(p.calculateInverseDynamics(
            robot, dof_pos, list(qd_np), zeros_dof))                # (14,)
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
        log["lc_dist_left"].append(dr_L)
        log["lc_dist_right"].append(dr_R)
        log["radius"].append(r_now)
        log["runtime_ms"].append((t1 - t0) * 1e3)

        if collision:
            print(f"[sim] t={sim_t:.3f}s  COLLISION  dist={collision_dist:.5f}"
                  f"  inter_arm={inter_dist:.5f}  Gamma={Gamma:+.4f}")
            break

        time.sleep(args.stepsize/3)

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
    csv_path = os.path.join(args.output_dir, f"dual_run_{int(time.time())}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
