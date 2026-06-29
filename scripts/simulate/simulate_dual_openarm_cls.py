#!/usr/bin/env python3
"""Dual-arm OpenArm simulation -- ORIGINAL VPP-TC TransformerGamma classifier.

Gamma here is the classifier LOGIT MARGIN (logit_safe - logit_unsafe), unitless,
NOT a distance in metres.  Threshold/eps are therefore in logit units (matching
upstream VPP-TC: threshold~2.5, eps~0.4).

Only the model + URDF differ from upstream VPP-TC; the QP / DS / dynamics /
collision-stop path is the original.

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

from vpptc.transformer_gamma_orig import TransformerGamma
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
        description="Dual-arm OpenArm sim (original VPP-TC TransformerGamma classifier)",
    )
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("--stepsize", type=float, default=1e-3)
    parser.add_argument("--realtime", action="store_true",
                        help="Each control period, sleep until wall-clock == sim time for 1:1 real-time viewing (only takes effect if fast enough)")
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--rand-init", action="store_true",
                        help="Start from a RANDOM collision-free joint config "
                             "(sampled within limits, seeded by --seed) instead "
                             "of the fixed Q_HOME_DUAL home pose.")
    parser.add_argument("--lc-center-left",  type=float, nargs=3,
                        default=[0.3, 0.03, 0.85])
    parser.add_argument("--lc-center-right", type=float, nargs=3,
                        default=[0.3, -0.03, 0.85])
    parser.add_argument("--lc-radius", type=float, default=0.1)
    parser.add_argument("--lc-omega",  type=float, default=2.0)
    parser.add_argument("--lc-plane-left",  type=str, default="xz",
                        choices=["xy", "xz", "yz"])
    parser.add_argument("--lc-plane-right", type=str, default="xy",
                        choices=["xy", "xz", "yz"])
    parser.add_argument("--gamma-threshold", type=float, default=5,
                        help="Trigger avoidance when predicted dist < this. "
                             "Default 20mm absorbs model bias (+4mm) and "
                             "low recall at 5mm.  Increase for safer / "
                             "more conservative behaviour.")
    parser.add_argument("--sca-eps",       type=float, default=0.1)
    parser.add_argument("--ctrl-dt", type=float, default=0.001,
                        help="control period (s): QP re-solved + torque "
                             "re-applied every round(ctrl_dt/stepsize) physics "
                             "steps (ZOH).  0.001=1kHz == per physics step, "
                             "matching the original dist sim.  Do NOT raise to "
                             "0.02 -- a 20ms ZOH lets the swing-in torque coast "
                             "and the arm self-collides.  The 20ms CBF/viability "
                             "horizon is SEPARATE (barrier dt=0.02, box-dt).")
    parser.add_argument("--box-dt", type=float, default=-1,
                        help="viability acceleration-box horizon (s). -1 = 0.02 "
                             "(the original dist-sim horizon; decoupled from the "
                             "control rate on purpose).")
    parser.add_argument("--sca-eps-decay", type=float, default=0.1)
    parser.add_argument("--sca-eps-floor", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1e-3,
                        help="QP regularisation weight")
    parser.add_argument("--model-path", type=str,
                        default=os.path.join(
                            _PROJECT_ROOT, "assets", "models",
                            "gamma_margin_tanh_d128.pt"))
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(_PROJECT_ROOT, "output"))
    parser.add_argument("--lc-kd",     type=float, default=200)
    parser.add_argument("--lc-kpos",   type=float, default=120.0)
    parser.add_argument("--lc-alpha",  type=float, default=30.0)
    parser.add_argument("--lc-kperp",  type=float, default=30.0)
    parser.add_argument("--no-gui", action="store_true",
                        help="Headless (DIRECT) mode")
    parser.add_argument("--record", type=str, default=None,
                        help="Path to an .mp4 file. Off-screen renders a frame "
                             "every --record-every steps (works in DIRECT mode) "
                             "and encodes the full sim to video.")
    parser.add_argument("--record-fps", type=int, default=30,
                        help="Output video frame rate.")
    parser.add_argument("--record-every", type=int, default=33,
                        help="Capture one frame every N sim steps "
                             "(33 @ 1ms step ~= 30fps real-time).")
    parser.add_argument("--record-size", type=int, nargs=2, default=[1280, 720],
                        help="Recorded frame WIDTH HEIGHT.")
    parser.add_argument("--log-danger", type=str, default=None,
                        help="Path to a .csv. Every step whose true inter-arm "
                             "distance < --danger-thresh, log (q, qd, "
                             "inter_dist, gamma) -- the deployment-distribution "
                             "hard negatives for DAgger re-labelling.")
    parser.add_argument("--danger-thresh", type=float, default=0.030,
                        help="Inter-arm distance (m) below which a step is "
                             "logged as a danger config (default 30mm).")
    parser.add_argument("--ctrl-vel-scale", type=float, default=1.0,
                        help="Scale the controller's velocity limit (viability "
                             "qd bound). Set to 0.1 to keep joint velocities in "
                             "the same range the model was trained on (vs0.1 "
                             "data), avoiding out-of-distribution gamma.")
    parser.add_argument("--ctrl-amax-scale", type=float, default=1.0,
                        help="Scale ONLY the controller's viability acceleration "
                             "bounds (not labels/qe). >1 gives the avoidance "
                             "controller more authority. Tests whether the "
                             "conservative a_max is crippling avoidance.")
    parser.add_argument("--debug-joints", action="store_true",
                        help="Every 200 steps print per-arm joint-limit "
                             "proximity / qdd-bound width / speed (to diagnose "
                             "an arm freezing at limits/singularity).")
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


def draw_circle_geom(center, radius, color, plane="xy", n_seg=48):
    """Draw the limit-cycle as a ring of small spheres (REAL geometry, so it
    shows up in off-screen getCameraImage renders, unlike addUserDebugLine).
    Plus a bigger sphere at the centre. Returns nothing; bodies are static."""
    i1, i2, _ = _PLANE_AXES[plane]
    rgba = list(color) + [1.0]
    vs = p.createVisualShape(p.GEOM_SPHERE, radius=0.006, rgbaColor=rgba)
    for i in range(n_seg):
        a = 2 * math.pi * i / n_seg
        pt = list(center)
        pt[i1] = center[i1] + radius * math.cos(a)
        pt[i2] = center[i2] + radius * math.sin(a)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=vs,
                          basePosition=pt)
    vc = p.createVisualShape(p.GEOM_SPHERE, radius=0.016,
                             rgbaColor=list(color) + [0.9])
    p.createMultiBody(baseMass=0, baseVisualShapeIndex=vc,
                      basePosition=list(center))


def find_link_index(robot_id, name):
    for i in range(p.getNumJoints(robot_id)):
        if p.getJointInfo(robot_id, i)[12].decode("utf-8") == name:
            return i
    raise ValueError(f"Link {name!r} not found")


def compute_gamma_and_grad(model, q_14, qd_14, threshold, device):
    """TransformerGamma: returns (logits, gamma); we use gamma and its gradient
    w.r.t. [q, qd].  Gradient only computed when threshold is exceeded, to avoid
    the autograd overhead in the safe-region case.
    """
    q_t = torch.tensor(q_14, dtype=torch.float32).unsqueeze(0)
    qd_t = torch.tensor(qd_14, dtype=torch.float32).unsqueeze(0)
    x = torch.cat([q_t, qd_t], dim=1).to(device)
    x.requires_grad_(True)
    _logits, gamma = model(x)          # classifier returns (logits, gamma)
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
    acc_max  = acc_max * args.ctrl_amax_scale   # controller-only authority scale
    qd_lim   = np.array(DUAL_VELOCITY_LIMITS, dtype=np.float32) * args.ctrl_vel_scale
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
        # Random collision-free start: perturb the safe home pose by per-joint
        # noise (full-range uniform almost always self-collides for a bimanual
        # arm).  Seeded by --seed; noise shrinks on retry until collision-free.
        rng = np.random.default_rng(args.seed)
        home = np.array(Q_HOME_DUAL)
        q0 = None
        for _try in range(400):
            scale = 0.9 * (0.97 ** _try)        # ~52deg, shrinking on retry
            cand = np.clip(home + rng.uniform(-scale, scale, size=N_DOF),
                           q_min_hw, q_max_hw)
            for jid, angle in zip(movable_joints, cand):
                p.resetJointState(robot, jid, float(angle))
            p.performCollisionDetection()
            # use getContactPoints (respects collision filters & reports only
            # real contacts) -- same definition the sim loop uses for collision.
            contacts = p.getContactPoints(bodyA=robot, bodyB=robot)
            if not any(c[8] < -5e-4 for c in contacts):  # <0.5mm pen = ok
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

    # Limit-cycle circles as REAL geometry so off-screen recording shows them
    # (addUserDebugLine above is invisible to getCameraImage).
    if args.record and args.no_gui:
        draw_circle_geom(lc_center_L, args.lc_radius,
                         color=[0, 0.5, 1], plane=args.lc_plane_left)
        draw_circle_geom(lc_center_R, args.lc_radius,
                         color=[1, 0.3, 0], plane=args.lc_plane_right)

    # ----- Load model -----
    device = torch.device("cpu")
    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    gamma_model = TransformerGamma(**cfg).to(device).eval()
    gamma_model.load_state_dict(ckpt["state_dict"])
    print(f"  [model] TransformerGamma  config={cfg}")
    print(f"  [sca]   threshold={args.gamma_threshold:.2f} (logit margin)  "
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
    _name2idx = {}
    for i in range(p.getNumJoints(robot)):
        name = p.getJointInfo(robot, i)[12].decode("utf-8")
        _name2idx[name] = i
        if name.startswith("openarm_left_"):
            left_body_links.append(i)
        elif name.startswith("openarm_right_"):
            right_body_links.append(i)
    from vpptc.blacklist_openarm import BLACKLIST_NAME_PAIRS
    bl_set = set()
    for na, nb in BLACKLIST_NAME_PAIRS:
        if na in _name2idx and nb in _name2idx:
            ia, ib = _name2idx[na], _name2idx[nb]
            bl_set.add((min(ia, ib), max(ia, ib)))

    log = {k: [] for k in [
        "time", "collision_dist", "inter_arm_dist", "gamma",
        "lc_dist_left", "lc_dist_right", "runtime_ms", "avoidance_on",
    ]}

    if not args.no_gui:
        time.sleep(2)

    # ----- Off-screen video recorder (works in DIRECT mode) -----
    recorder = None
    if args.record:
        import cv2
        rw, rh = args.record_size
        view_mat = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.0, 0.1, 0.95], distance=1.05,
            yaw=60, pitch=-22, roll=0, upAxisIndex=2)
        proj_mat = p.computeProjectionMatrixFOV(
            fov=60, aspect=rw / rh, nearVal=0.1, farVal=100.0)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(args.record, fourcc, args.record_fps, (rw, rh))
        recorder = {"cv2": cv2, "vw": vw, "view": view_mat, "proj": proj_mat,
                    "w": rw, "h": rh, "n": 0}
        print(f"  [record] -> {args.record}  {rw}x{rh}@{args.record_fps}fps  "
              f"(1 frame / {args.record_every} steps)")

    # Live HUD state, overlaid on each recorded frame.
    hud = {"t": 0.0, "gamma": 0.0, "thr": args.gamma_threshold,
           "on": False, "dmm": float("nan")}

    def _grab_frame():
        if recorder is None:
            return
        cv2 = recorder["cv2"]
        img = p.getCameraImage(
            recorder["w"], recorder["h"],
            viewMatrix=recorder["view"], projectionMatrix=recorder["proj"],
            renderer=p.ER_TINY_RENDERER)
        rgb = np.reshape(img[2], (recorder["h"], recorder["w"], 4))[:, :, :3]
        bgr = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        # ---- HUD overlay ----
        on = hud["on"]
        lines = [
            (f"t = {hud['t']:5.2f} s", (255, 255, 255)),
            (f"Gamma = {hud['gamma']:+6.2f}  (thr {hud['thr']:.1f})",
             (80, 220, 80) if hud["gamma"] > hud["thr"] else (80, 140, 255)),
            (f"avoidance: {'ON' if on else 'off'}",
             (60, 90, 255) if on else (180, 180, 180)),
            (f"inter-arm min: {hud['dmm']:+6.2f} mm", (255, 255, 255)),
        ]
        y = 34
        for txt, col in lines:
            cv2.putText(bgr, txt, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 0, 0), 4, cv2.LINE_AA)        # black outline
            cv2.putText(bgr, txt, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        col, 2, cv2.LINE_AA)
            y += 34
        recorder["vw"].write(bgr)
        recorder["n"] += 1

    wall_start = time.time()
    # Controller runs at the CBF horizon (CTRL_DT=0.02); physics integrates at
    # args.stepsize.  Zero-order-hold the torque over DECIM physics steps so the
    # control period == the CBF dt (fixes the dt-mismatch: CBF planned 20ms but
    # the world advanced 1ms, making predicted dgamma ~20x the actual).
    CTRL_DT = args.ctrl_dt
    DECIM = max(1, round(CTRL_DT / args.stepsize))
    EPS_SCALE = CTRL_DT / 0.02   # keep "barrier strength per unit time" constant
    BOX_DT = args.box_dt if args.box_dt > 0 else 0.02   # viability-box horizon (decoupled from ctrl rate)
    print(f"  [ctrl] CTRL_DT={CTRL_DT}s  stepsize={args.stepsize}s  "
          f"decimation={DECIM} physics steps / control solve")
    num_steps = int(args.duration / (DECIM * args.stepsize))
    sim_t = 0.0
    zeros_all = [0.0] * n_all_dof
    n_avoidance = 0
    n_soft = 0
    danger_rows = []   # (q14, qd14, inter_dist, gamma) when inter_dist<thresh

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

        qdd_lb, qdd_ub = compute_joint_acceleration_bounds_vec(
            q_np, qd_np, q_min_hw, q_max_hw, qd_lim, acc_max,
            dt=BOX_DT, viability=True)

        if args.debug_joints and step_i % 200 == 0:
            # split 0:7 = left, 7:14 = right (DUAL = LEFT + RIGHT)
            for nm, sl in (("L", slice(0, 7)), ("R", slice(7, 14))):
                qa, qda = q_np[sl], qd_np[sl]
                lo, hi = q_min_hw[sl], q_max_hw[sl]
                margin = np.minimum(qa - lo, hi - qa)          # rad to nearest limit
                width = (qdd_ub[sl] - qdd_lb[sl])              # accel freedom
                n_at_lim = int((margin < np.radians(3)).sum())
                tight = int((width < 1.0).sum())               # near-frozen joints
                print(f"      [dbg {nm}] |qd|={np.linalg.norm(qda):5.2f}  "
                      f"min_lim_margin={np.degrees(margin.min()):5.1f}deg  "
                      f"joints@limit(<3deg)={n_at_lim}  "
                      f"qdd_width<1: {tight}/7  ee_speed="
                      f"{np.linalg.norm(left_vel if nm=='L' else right_vel):.3f}")

        all_states = p.getJointStates(robot, all_movable_joints)
        all_pos = [s[0] for s in all_states]
        all_vel = [s[1] for s in all_states]

        # ---- Barrier gamma + gradient (learned logit) ----
        Gamma, grad_gamma = compute_gamma_and_grad(
            gamma_model, q, qd, args.gamma_threshold, device)
        if step_i % 100 == 0:
            print(f"[sim] t={sim_t:6.3f}s  Gamma={Gamma:+7.4f}  "
                  f"avoidance={'ON ' if grad_gamma is not None else 'off'}")

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
        _qdd_log = np.zeros(N_DOF)   # commanded joint accel (diag)
        _tau_log = np.zeros(N_DOF)   # commanded joint torque (diag)
        _tau_apply = [0.0] * N_DOF   # torque held over the decimation
        if grad_gamma is not None:
            n_avoidance += 1
            dt = 0.02   # CBF linearization horizon (design constant, ~invariant)
            gq  = grad_gamma[:N_DOF]
            gqd = grad_gamma[N_DOF:]
            g_eff = 0.5 * gq * dt ** 2 + gqd * dt
            c_const = gq.dot(qd_np) * dt
            eps = args.sca_eps   # raw eps (no dt scaling) -- as the user sets it
            constraints.append(
                g_eff @ M_inv @ u >= eps - c_const + g_eff @ M_inv @ tau_id
            )
            prob = cp.Problem(cp.Minimize(objective), constraints)
            try:
                prob.solve(solver=cp.OSQP)
            except cp.SolverError:
                soft = True
                print(f"[sim] t={sim_t:6.3f}s  OSQP SolverError "
                      f"(avoidance solve, eps={eps:.3f}) -> soft")

            while (not soft) and prob.status != cp.OPTIMAL and eps > args.sca_eps_floor:
                eps -= args.sca_eps_decay
                constraints[-1] = (
                    g_eff @ M_inv @ u >= eps - c_const + g_eff @ M_inv @ tau_id
                )
                prob = cp.Problem(cp.Minimize(objective), constraints)
                try:
                    prob.solve(solver=cp.OSQP)
                except cp.SolverError:
                    soft = True
                    print(f"[sim] t={sim_t:6.3f}s  OSQP SolverError "
                          f"(SCA retry, eps={eps:.3f}) -> soft")
                    break

            if soft or eps < args.sca_eps_floor or prob.status != cp.OPTIMAL:
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
                _qdd_log = np.array(qdd_cmd); _tau_log = np.array(tau_arm)
                _tau_apply = list(tau_arm)
        else:
            # Safe region (grad_gamma is None): plain QP, only accel-bound
            # constraints.  Mirror the avoidance branch's QP-failure handling
            # (try/except SolverError + non-OPTIMAL -> soft).  The upstream
            # VPP-TC repo left this branch unguarded; OSQP can still throw on a
            # numerically borderline solve, which crashes the loop.
            prob = cp.Problem(cp.Minimize(objective), constraints)
            try:
                prob.solve(solver=cp.OSQP)
            except cp.SolverError:
                soft = True
                print(f"[sim] t={sim_t:6.3f}s  OSQP SolverError "
                      f"(safe-region solve) -> soft")
            if not soft and prob.status != cp.OPTIMAL:
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
                _tau_apply = list(tau_arm)

        if not soft:
            tau_cmd = np.array(u.value)
            _qdd_log = M_inv @ (tau_cmd - tau_id)
            _tau_log = tau_cmd
            _tau_apply = tau_cmd.tolist()

        # zero-order hold: apply the control torque over DECIM physics steps
        for _ in range(DECIM):
            p.setJointMotorControlArray(
                robot, movable_joints, p.TORQUE_CONTROL, forces=_tau_apply)
            p.stepSimulation()

        sim_t += DECIM * args.stepsize
        if args.realtime:                               # real-time throttling: if running ahead, sleep until real time for 1:1 viewing
            _slack = (wall_start + sim_t) - time.time()
            if _slack > 0:
                time.sleep(_slack)
        t1 = time.perf_counter()

        # Self-collision distance (ground truth): min getClosestPoints over all
        # non-blacklist link pairs -- exactly the quantity the model is trained on
        # and the upstream VPP-TC collision check.  collision iff any pair < 0.
        inter_dist = math.inf
        inter_pair = None
        for pt in p.getClosestPoints(robot, robot, 2.0):
            a, b = pt[3], pt[4]
            if a == b or (min(a, b), max(a, b)) in bl_set:
                continue
            if pt[8] < inter_dist:
                inter_dist = pt[8]
                inter_pair = (a, b)
        collision = inter_dist < 0
        collision_dist = inter_dist

        # DAgger: log deployment-distribution hard negatives (q, qd) wherever
        # the arms get within --danger-thresh, for proper re-labelling later.
        if args.log_danger is not None and inter_dist < args.danger_thresh:
            danger_rows.append(list(q_np) + list(qd_np)
                               + [inter_dist, Gamma])

        log["time"].append(sim_t)
        log["collision_dist"].append(collision_dist)
        log["inter_arm_dist"].append(inter_dist)
        log["gamma"].append(Gamma)
        log["lc_dist_left"].append(dr_L)
        log["lc_dist_right"].append(dr_R)
        log["runtime_ms"].append((t1 - t0) * 1e3)
        log["avoidance_on"].append(1 if grad_gamma is not None else 0)

        if recorder is not None and step_i % args.record_every == 0:
            hud["t"] = sim_t
            hud["gamma"] = Gamma
            hud["on"] = grad_gamma is not None
            hud["dmm"] = inter_dist * 1000.0
            _grab_frame()

        if collision:
            _lname = {i: p.getJointInfo(robot, i)[12].decode()
                      for i in range(p.getNumJoints(robot))}
            _cpair = (f"{_lname.get(inter_pair[0], inter_pair[0])} <-> "
                      f"{_lname.get(inter_pair[1], inter_pair[1])}"
                      if inter_pair else "?")
            print(f"\n[sim] t={sim_t:.3f}s  COLLISION  "
                  f"dist={inter_dist*1000:+.2f}mm  pair=[{_cpair}]  "
                  f"Gamma(logit)={Gamma:+.3f}")
            if recorder is not None:
                # hold the collision frame ~1s so it's visible in the video
                hud["t"] = sim_t
                hud["gamma"] = Gamma
                hud["on"] = grad_gamma is not None
                hud["dmm"] = inter_dist * 1000.0
                for _ in range(args.record_fps):
                    _grab_frame()
            break

        if not args.no_gui:
            # Real-time pacing: sleep so wall-clock tracks sim_t (1 sim-second =
            # 1 wall-second) when compute keeps up; best-effort (no sleep) when
            # the loop is already behind real-time.
            behind = sim_t - (time.time() - wall_start)
            if behind > 0:
                time.sleep(behind)

    if recorder is not None:
        recorder["vw"].release()
        print(f"  [record] wrote {recorder['n']} frames -> {args.record}")

    # --- Summary ---
    elapsed = time.time() - wall_start
    runtimes = np.array(log["runtime_ms"][1:])
    n_steps = len(log["time"])
    print(f"\n{'=' * 60}")
    print(f"Simulation finished in {elapsed:.2f}s wall-clock")
    print(f"  Steps           : {n_steps}")
    print(f"  Avoidance on    : {n_avoidance}/{n_steps}  "
          f"({100*n_avoidance/max(1,n_steps):.1f}%)")
    _lcl = np.abs(np.array(log["lc_dist_left"])); _lcr = np.abs(np.array(log["lc_dist_right"]))
    _h = len(_lcl) // 2
    print(f"  Track radial err: L={_lcl[_h:].mean()*1000:.1f}mm R={_lcr[_h:].mean()*1000:.1f}mm "
          f"(2nd-half mean |radial err|; low=traces circle, high=braking/not tracking)")
    print(f"  Soft fallback   : {n_soft}/{max(1,n_avoidance)}  "
          f"(QP infeasible)")
    if len(runtimes) > 0:
        print(f"  Mean step       : {runtimes.mean():.3f} ms")
        print(f"  Median step     : {np.median(runtimes):.3f} ms")
        print(f"  95th pct        : {np.percentile(runtimes, 95):.3f} ms")
        print(f"  Max step        : {runtimes.max():.3f} ms")

    # NOTE: Gamma here is a logit margin (unitless), NOT metres, so the
    # regressor's predicted-vs-true-distance RMSE is meaningless for this
    # classifier and is omitted.  Report only the true closest approach.
    d_arr = np.array(log["inter_arm_dist"])
    d_v = d_arr[np.isfinite(d_arr)]
    if len(d_v) > 100:
        print(f"\n  True closest approach (inter-arm):")
        print(f"    min true d  : {d_v.min()*1000:+.2f}mm  "
              f"(closest the arms came; <0 = collided)")

    # ----- Gamma summary -----
    g_arr = np.array(log["gamma"], dtype=float)
    thr = args.gamma_threshold
    if len(g_arr) > 0:
        below = (g_arr <= thr)
        # time near the barrier (within +1 logit of threshold = "held at wall")
        near = np.abs(g_arr - thr) < 1.0
        print(f"\n  Gamma (logit margin)  [threshold {thr:.2f}]:")
        print(f"    min / mean / max : {g_arr.min():+.2f} / "
              f"{g_arr.mean():+.2f} / {g_arr.max():+.2f}")
        print(f"    <= threshold     : {below.mean()*100:.1f}% of steps "
              f"(avoidance triggered)")
        print(f"    within +-1 of thr: {near.mean()*100:.1f}% of steps "
              f"(barrier 'holding the wall')")
        print(f"    min margin to thr: {(g_arr.min()-thr):+.2f}  "
              f"(how far Gamma slipped past the barrier; <0 = breached)")

    # ----- DAgger: dump deployment-distribution hard negatives -----
    if args.log_danger is not None:
        import csv as _csv
        hdr = ([f"joint_{i}_pos" for i in range(14)]
               + [f"joint_{i}_vel" for i in range(14)]
               + ["sim_inter_dist", "gamma_logit"])
        write_header = not os.path.exists(args.log_danger)
        with open(args.log_danger, "a", newline="") as f:
            w = _csv.writer(f)
            if write_header:
                w.writerow(hdr)
            w.writerows(danger_rows)
        print(f"\n  [log-danger] appended {len(danger_rows)} rows "
              f"(inter_dist < {args.danger_thresh*1000:.0f}mm) -> {args.log_danger}")



if __name__ == "__main__":
    main()
