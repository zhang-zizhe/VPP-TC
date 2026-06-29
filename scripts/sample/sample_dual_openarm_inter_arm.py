#!/usr/bin/env python3
"""Targeted sampler: biased toward INTER-ARM interaction configurations.

Uniform 14-D sampling almost never produces poses where the two arms are
close to each other, so uniform-trained models do poorly on inter-arm
collisions (esp. arm-finger and finger-finger between left and right hand).
This sampler generates configs where both arms occupy the same region of
3D workspace, then records min self-distance the same way as
sample_dual_openarm_dist.py.

Strategies (mixed per-sample):
  70%  CLOSE  : sample shared target P, IK both arms to P + δ (|δ| ≤ 10cm)
  30%  MEDIUM : independent targets P_L, P_R in shared workspace box,
                |P_L - P_R| ≤ 30 cm
After IK, add Gaussian joint noise σ ~ U[0, 0.3] rad (clipped to limits)
to spread coverage from exactly-touching to merely-nearby.

Output CSV adds two extra columns (offender_a, offender_b) with the
link names of the closest pair at q, for downstream diagnosis.

Usage
-----
    python scripts/sample/sample_dual_openarm_inter_arm.py \
        --n-samples 2000000 --workers 16 \
        --output-name openarm_dual_inter_arm.csv
"""

import argparse
import csv
import multiprocessing as mp
import os
import re
import sys
import time

import numpy as np
import pybullet as p
import pybullet_data

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, os.pardir, os.pardir)
sys.path.insert(0, os.path.abspath(_PROJECT_ROOT))

from vpptc.utils_openarm import (DUAL_ACCELERATION_LIMITS, DUAL_POS_LIMITS,
                                 OPENARM_VELOCITY_LIMITS)
from vpptc.utils import compute_qe


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-samples", type=int, default=2_000_000)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--mode", type=str, default="targeted",
                    choices=["targeted", "random"],
                    help="targeted = IK-biased inter-arm boundary sampling; "
                         "random = fully uniform (q, qd), accept everything")
    ap.add_argument("--clip-min", type=float, default=-0.05,
                    help="(DEPRECATED / unused) labels are no longer clipped")
    ap.add_argument("--urdf", type=str,
                    default=os.path.join(
                        _PROJECT_ROOT, "assets", "urdf", "openarm_description",
                        "urdf", "robot", "openarm_bimanual.urdf"))
    ap.add_argument("--output-dir", type=str,
                    default=os.path.join(_PROJECT_ROOT, "output"))
    ap.add_argument("--output-name", type=str,
                    default="openarm_dual_inter_arm.csv")
    ap.add_argument("--frac-close", type=float, default=0.20,
                    help="Fraction of samples using shared-target strategy")
    ap.add_argument("--close-offset", type=float, default=0.18,
                    help="Max |δ| for left/right targets around shared P (m)")
    ap.add_argument("--med-min-sep", type=float, default=0.12,
                    help="Min |P_L - P_R| in MEDIUM strategy (m)")
    ap.add_argument("--med-max-sep", type=float, default=0.32,
                    help="Max |P_L - P_R| in MEDIUM strategy (m)")
    ap.add_argument("--noise-sigma-max", type=float, default=0.12,
                    help="Max σ of joint noise added on IK solution (rad)")
    ap.add_argument("--require-inter-arm", type=int, default=1,
                    help="1 = reject samples whose offender is not inter-arm")
    ap.add_argument("--label-min", type=float, default=-0.025,
                    help="Reject sample if d_q < this (m); set -1 to disable")
    ap.add_argument("--label-max", type=float, default=0.040,
                    help="Reject sample if d_q > this (m); set 1 to disable")
    ap.add_argument("--max-retries", type=int, default=50,
                    help="Max retries per sample")
    ap.add_argument("--seed", type=int, default=20260529)
    ap.add_argument("--vel-scale", type=float, default=1.0,
                    help="Scale the sampled velocity range. OpenArm's official "
                         "MoveIt uses default_velocity_scaling_factor=0.1, i.e. "
                         "operates at 10%% of the no-load max speeds. Pass 0.1 to "
                         "sample realistic operating velocities (the URDF "
                         "max_velocity values are no-load maxes).")
    return ap.parse_args()


# ======================================================================
# Per-process PyBullet setup
# ======================================================================

def _setup(urdf_path):
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
    robot = p.loadURDF(urdf_path, useFixedBase=True, flags=flags,
                       physicsClientId=cid)

    # Disable link5 <-> link7 within each arm
    link5s, link7s = [], []
    for i in range(p.getNumJoints(robot, physicsClientId=cid)):
        name = p.getJointInfo(robot, i, physicsClientId=cid)[12].decode()
        if name.endswith("_link5"): link5s.append(i)
        elif name.endswith("_link7"): link7s.append(i)
    for l5 in link5s:
        for l7 in link7s:
            if abs(l5 - l7) < 5:
                p.setCollisionFilterPair(robot, robot, l5, l7,
                                         enableCollision=0, physicsClientId=cid)

    # Discover arm joints (14), finger joints, EE links, shoulder positions
    arm_re = re.compile(r"openarm_(left|right)_joint[1-7]$")
    arm_joints_L, arm_joints_R = [], []
    pos_lims_L, pos_lims_R = [], []
    vel_lims_L, vel_lims_R = [], []
    finger_joints = []
    name_to_idx, idx_to_name = {}, {-1: "base"}
    n_total_joints = p.getNumJoints(robot, physicsClientId=cid)
    for i in range(n_total_joints):
        info = p.getJointInfo(robot, i, physicsClientId=cid)
        jname = info[1].decode()
        lname = info[12].decode()
        name_to_idx[lname] = i
        idx_to_name[i] = lname
        if "finger_joint" in jname:
            finger_joints.append(i)
            p.resetJointState(robot, i, 0.01, physicsClientId=cid)
        if info[2] == p.JOINT_REVOLUTE:
            m = arm_re.match(jname)
            if m:
                if m.group(1) == "left":
                    arm_joints_L.append(i)
                    pos_lims_L.append((info[8], info[9]))
                    vel_lims_L.append((-info[11], info[11]))
                else:
                    arm_joints_R.append(i)
                    pos_lims_R.append((info[8], info[9]))
                    vel_lims_R.append((-info[11], info[11]))

    # EE link indices for IK
    ee_L = name_to_idx.get("openarm_left_hand", None)
    ee_R = name_to_idx.get("openarm_right_hand", None)
    if ee_L is None:
        # Fallback: last link of left arm
        ee_L = max(arm_joints_L)
    if ee_R is None:
        ee_R = max(arm_joints_R)

    # Shoulder positions (link0 of each arm) for workspace box
    p.stepSimulation(physicsClientId=cid)
    sh_L = np.array(p.getLinkState(robot, name_to_idx["openarm_left_link0"],
                                    physicsClientId=cid)[0])
    sh_R = np.array(p.getLinkState(robot, name_to_idx["openarm_right_link0"],
                                    physicsClientId=cid)[0])

    # Blacklist
    from vpptc.blacklist_openarm import BLACKLIST_NAME_PAIRS
    blacklist = set()
    for na, nb in BLACKLIST_NAME_PAIRS:
        if na in name_to_idx and nb in name_to_idx:
            ia, ib = name_to_idx[na], name_to_idx[nb]
            blacklist.add((min(ia, ib), max(ia, ib)))
    for na, nb in [("openarm_left_link0","openarm_right_link0"),
                   ("openarm_left_link0","openarm_right_link1"),
                   ("openarm_left_link1","openarm_right_link0"),
                   ("openarm_left_link1","openarm_right_link1")]:
        if na in name_to_idx and nb in name_to_idx:
            ia, ib = name_to_idx[na], name_to_idx[nb]
            blacklist.add((min(ia, ib), max(ia, ib)))

    return {
        "cid": cid, "robot": robot,
        "arm_L": arm_joints_L, "arm_R": arm_joints_R,
        "pos_lims_L": pos_lims_L, "pos_lims_R": pos_lims_R,
        "vel_lims_L": vel_lims_L, "vel_lims_R": vel_lims_R,
        "finger": finger_joints,
        "ee_L": ee_L, "ee_R": ee_R,
        "sh_L": sh_L, "sh_R": sh_R,
        "blacklist": blacklist,
        "n_total_joints": n_total_joints,
        "name_to_idx": name_to_idx, "idx_to_name": idx_to_name,
    }


# Fixed getClosestPoints search radius. Set well beyond the robot's own
# kinematic span (~1.5 m), so EVERY non-blacklisted link pair is returned
# and the reported min distance is ALWAYS the exact true value -- no cap on
# positive distances, no cap on penetration. (Any radius >= the arm span is
# functionally infinite for self-distance, which structurally maxes far below
# this.)
_SEARCH_RADIUS = 10.0


def _min_self_dist(env, q14):
    """Compute min distance and offender pair at config q14 (14 arm joints).

    q14 ordering: [left arm 7, right arm 7] matching arm_L + arm_R.
    Returns (d_min, name_a, name_b). The distance is the EXACT true min
    self-distance (no clipping at any radius); penetration is negative.
    """
    cid, robot = env["cid"], env["robot"]
    arm = env["arm_L"] + env["arm_R"]
    for jid, angle in zip(arm, q14):
        p.resetJointState(robot, jid, angle, physicsClientId=cid)
    for fj in env["finger"]:
        p.resetJointState(robot, fj, 0.01, physicsClientId=cid)
    p.stepSimulation(physicsClientId=cid)

    pts = p.getClosestPoints(bodyA=robot, bodyB=robot,
                             distance=_SEARCH_RADIUS, physicsClientId=cid)
    if not pts:
        return _SEARCH_RADIUS, "", ""
    best_d, best_a, best_b = float("inf"), -1, -1
    for pt in pts:
        a, b = pt[3], pt[4]
        if a == b:
            continue
        if (min(a, b), max(a, b)) in env["blacklist"]:
            continue
        if pt[8] < best_d:
            best_d, best_a, best_b = pt[8], a, b
    return best_d, env["idx_to_name"].get(best_a, ""), env["idx_to_name"].get(best_b, "")


def _ik_arm(env, ee_link, target_pos, lock_arm_joints, lock_q, rng):
    """IK ee_link to target_pos while holding lock_arm_joints at lock_q.

    Returns the 7 joint values for the arm whose EE is ee_link (the arm
    we're solving), in arm-joint order.
    """
    cid, robot = env["cid"], env["robot"]
    n_total = env["n_total_joints"]
    arm_L, arm_R = env["arm_L"], env["arm_R"]
    target_arm = arm_L if ee_link == env["ee_L"] else arm_R

    # Lock the OTHER arm by setting joint state (IK uses rest configuration
    # via current joint states for damped least squares).
    for jid, val in zip(lock_arm_joints, lock_q):
        p.resetJointState(robot, jid, val, physicsClientId=cid)
    # Random init for the arm we're solving, to diversify IK solutions.
    for jid in target_arm:
        info = p.getJointInfo(robot, jid, physicsClientId=cid)
        p.resetJointState(robot, jid,
                          rng.uniform(info[8], info[9]),
                          physicsClientId=cid)

    sol = p.calculateInverseKinematics(
        robot, ee_link, target_pos,
        maxNumIterations=40, residualThreshold=1e-3,
        physicsClientId=cid,
    )
    # sol has length = number of movable joints; map joint index -> position
    # in sol by counting movable joints up to that index.
    movable_idx = []
    for i in range(n_total):
        if p.getJointInfo(robot, i, physicsClientId=cid)[2] != p.JOINT_FIXED:
            movable_idx.append(i)
    pos_in_sol = {j: k for k, j in enumerate(movable_idx)}
    return np.array([sol[pos_in_sol[j]] for j in target_arm])


def _sample_config(env, rng, args):
    """Return q14 array (14,) sampled with biased strategy."""
    pos_lims_L = np.array(env["pos_lims_L"])
    pos_lims_R = np.array(env["pos_lims_R"])
    arm_L, arm_R = env["arm_L"], env["arm_R"]
    ee_L, ee_R = env["ee_L"], env["ee_R"]
    sh_L, sh_R = env["sh_L"], env["sh_R"]

    # Shared workspace box: union of the two shoulder neighborhoods.
    # Both shoulders are at base level; arms reach ~0.7 m. The interesting
    # region for inter-arm collision is roughly between/in-front-of them.
    sh_mid = 0.5 * (sh_L + sh_R)
    sep = np.linalg.norm(sh_L - sh_R)         # shoulder separation
    reach = 0.70                              # rough max reach per arm
    # box centered at shoulder midpoint, half-extents:
    box_half = np.array([
        sep * 0.6,               # x: ~within shoulder span, force arms to meet
        0.25,                    # y: forward extent (away from torso)
        0.40,                    # z: shoulder height ± 40cm
    ])
    box_center = sh_mid + np.array([0.0, 0.25, 0.0])  # push forward, off body

    strategy = "close" if rng.random() < args.frac_close else "medium"

    if strategy == "close":
        P = box_center + rng.uniform(-box_half, box_half)
        delta_L = rng.uniform(-args.close_offset, args.close_offset, size=3)
        delta_R = rng.uniform(-args.close_offset, args.close_offset, size=3)
        target_L = P + delta_L
        target_R = P + delta_R
    else:
        # independent targets within box, clamp separation
        target_L = box_center + rng.uniform(-box_half, box_half)
        target_R = box_center + rng.uniform(-box_half, box_half)
        sep_vec = target_R - target_L
        sep_norm = np.linalg.norm(sep_vec)
        if sep_norm > args.med_max_sep:
            target_R = target_L + sep_vec * (args.med_max_sep / sep_norm)
        elif sep_norm < args.med_min_sep and sep_norm > 1e-6:
            target_R = target_L + sep_vec * (args.med_min_sep / sep_norm)

    # IK both arms. Solve left first with right at random init, then solve
    # right with left locked at its solution.
    q_R_init = rng.uniform(pos_lims_R[:, 0], pos_lims_R[:, 1])
    q_L = _ik_arm(env, ee_L, target_L, arm_R, q_R_init, rng)
    q_L = np.clip(q_L, pos_lims_L[:, 0], pos_lims_L[:, 1])
    q_R = _ik_arm(env, ee_R, target_R, arm_L, q_L, rng)
    q_R = np.clip(q_R, pos_lims_R[:, 0], pos_lims_R[:, 1])

    # Joint noise σ ~ U[0, σ_max]
    sigma = rng.uniform(0.0, args.noise_sigma_max)
    q_L = q_L + rng.normal(0, sigma, size=7)
    q_R = q_R + rng.normal(0, sigma, size=7)
    q_L = np.clip(q_L, pos_lims_L[:, 0], pos_lims_L[:, 1])
    q_R = np.clip(q_R, pos_lims_R[:, 0], pos_lims_R[:, 1])

    return np.concatenate([q_L, q_R])


# ======================================================================
# Worker
# ======================================================================

_shared_counter = None
def _worker_init(counter):
    global _shared_counter
    _shared_counter = counter


def _worker(task):
    n_target, seed, urdf_path, clip_min, args = task
    env = _setup(urdf_path)
    rng = np.random.default_rng(seed)

    # Velocity range from the OPENARM_VELOCITY_LIMITS constant (single source of
    # truth = current operating limit, 0.3x no-load), NOT the URDF info[11]
    # (which is the no-load max).  Keeps sampling velocity consistent with the
    # controller's qd bound.
    _VL = np.array([[-v, v] for v in OPENARM_VELOCITY_LIMITS])
    vel_lims_L = _VL
    vel_lims_R = _VL

    pos_lims_L = np.array(env["pos_lims_L"])
    pos_lims_R = np.array(env["pos_lims_R"])

    out_q, out_v, out_qe = [], [], []
    out_dq, out_dqe, out_lab = [], [], []
    out_oa, out_ob = [], []
    local_acc = 0
    ik_fail = 0
    total_attempts = 0

    def _arm_of(name):
        # "openarm_left_link3" -> "L"; "openarm_right_left_finger" -> "R"
        if name.startswith("openarm_left_"):
            return "L"
        if name.startswith("openarm_right_"):
            return "R"
        return "X"  # body, base, unknown

    def _is_inter_arm(a, b):
        aa, ab = _arm_of(a), _arm_of(b)
        return (aa == "L" and ab == "R") or (aa == "R" and ab == "L")

    def _rand_q():
        return np.concatenate([
            rng.uniform(pos_lims_L[:, 0], pos_lims_L[:, 1]),
            rng.uniform(pos_lims_R[:, 0], pos_lims_R[:, 1]),
        ])

    def _rand_v():
        vs = args.vel_scale
        return np.concatenate([
            rng.uniform(vel_lims_L[:, 0] * vs, vel_lims_L[:, 1] * vs),
            rng.uniform(vel_lims_R[:, 0] * vs, vel_lims_R[:, 1] * vs),
        ])

    def _eval_qe(q, v, d_q):
        """Compute braking stop pose qe and viability label, NO clipping.

        Returns (qe, d_qe, d_lab) where d_lab = min(d_q, d_qe).
        Only called AFTER a pose passes the targeted filter, so the
        (relatively expensive) compute_qe + 2nd collision query is not
        wasted on the ~95% of rejected attempts.
        """
        try:
            qe = np.asarray(compute_qe(
                q.tolist(), v.tolist(),
                acc_limits=DUAL_ACCELERATION_LIMITS,
                pos_limits=DUAL_POS_LIMITS), dtype=float)
        except Exception:
            qe = q.copy()
        d_qe, _, _ = _min_self_dist(env, qe)
        return qe, d_qe, min(d_q, d_qe)

    # label filter bounds (disabled when label-min<=-1 / label-max>=1)
    lo = -1e9 if args.label_min <= -1 else args.label_min
    hi = 1e9 if args.label_max >= 1 else args.label_max

    for _ in range(n_target):
        if args.mode == "random":
            # fully uniform (q, qd); accept everything, no filter, no clip
            total_attempts += 1
            q = _rand_q()
            v = _rand_v()
            d_q, na, nb = _min_self_dist(env, q)
            qe, d_qe, d_lab = _eval_qe(q, v, d_q)
            accepted = True
        else:
            # targeted: IK-biased inter-arm, filter to viability boundary.
            # Cheap path first: IK + ONE collision query for d_q/offender.
            # Only compute qe (compute_qe + 2nd query) once the pose passes.
            accepted = False
            for _retry in range(max(1, args.max_retries)):
                total_attempts += 1
                try:
                    q = _sample_config(env, rng, args)
                except Exception:
                    ik_fail += 1
                    q = _rand_q()
                d_q, na, nb = _min_self_dist(env, q)
                ok_pair = (not args.require_inter_arm) or _is_inter_arm(na, nb)
                # Filter on the GEOMETRIC pose proximity d_q (the inter-arm
                # boundary), NOT d_lab. Velocity stays fully random; the label
                # min(d_q,d_qe) is then recorded unfiltered (deep qe kept).
                ok_label = (lo <= d_q <= hi)
                if ok_pair and ok_label:
                    v = _rand_v()
                    qe, d_qe, d_lab = _eval_qe(q, v, d_q)
                    accepted = True
                    break
            if not accepted:
                continue

        out_q.append(q); out_v.append(v); out_qe.append(qe)
        out_dq.append(d_q); out_dqe.append(d_qe); out_lab.append(d_lab)
        out_oa.append(na); out_ob.append(nb)
        local_acc += 1

        if local_acc % 500 == 0 and _shared_counter is not None:
            with _shared_counter.get_lock():
                _shared_counter.value += 500

    if local_acc % 500 != 0 and _shared_counter is not None:
        with _shared_counter.get_lock():
            _shared_counter.value += local_acc % 500

    if ik_fail:
        print(f"  [worker seed={seed}] IK fallback x{ik_fail}", flush=True)
    print(f"  [worker seed={seed}] {local_acc} kept / {total_attempts} attempts "
          f"= {100*local_acc/max(total_attempts,1):.1f}% hit rate", flush=True)

    p.disconnect(physicsClientId=env["cid"])
    return (np.array(out_q), np.array(out_v), np.array(out_qe),
            np.array(out_dq), np.array(out_dqe), np.array(out_lab),
            np.array(out_oa), np.array(out_ob))


# ======================================================================
# Parallel runner
# ======================================================================

def run_parallel(tasks, n_target, n_workers):
    report_every = max(1, n_target // 50)
    width = len(str(n_target))
    next_report = report_every
    t0 = time.time()
    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    with ctx.Pool(n_workers, initializer=_worker_init,
                  initargs=(counter,)) as pool:
        ar = pool.map_async(_worker, tasks)
        while not ar.ready():
            ar.wait(timeout=0.5)
            n = counter.value
            if n >= next_report:
                elapsed = time.time() - t0
                rate = n / max(elapsed, 1e-6)
                eta_min = (n_target - n) / max(rate, 1e-6) / 60.0
                print(f"  [{min(n, n_target):>{width}}/{n_target}]  "
                      f"elapsed={elapsed:.1f}s  rate={rate:.1f}/s  "
                      f"eta={eta_min:.1f} min")
                next_report = (n // report_every + 1) * report_every
        results = ar.get()
    print(f"  Done in {time.time() - t0:.1f}s")
    return results


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    n_workers = max(1, min(args.workers, args.n_samples))
    print(f"OpenArm INTER-ARM targeted sampling")
    print(f"  Total samples : {args.n_samples}")
    print(f"  Workers       : {n_workers}")
    print(f"  frac CLOSE    : {args.frac_close}")
    print(f"  close offset  : {args.close_offset} m")
    print(f"  med max sep   : {args.med_max_sep} m")
    print(f"  noise σ max   : {args.noise_sigma_max} rad")
    print()

    base, rem = divmod(args.n_samples, n_workers)
    counts = [base + (1 if i < rem else 0) for i in range(n_workers)]
    rng_root = np.random.SeedSequence(args.seed)
    seeds = rng_root.spawn(n_workers)
    tasks = [(counts[i], seeds[i], args.urdf, args.clip_min, args)
             for i in range(n_workers)]

    results = run_parallel(tasks, args.n_samples, n_workers)

    # New 8-element return: q, v, qe, dq, dqe, lab, oa, ob
    results = [r for r in results if len(r[0])]  # drop empty workers
    all_q   = np.concatenate([r[0] for r in results])
    all_v   = np.concatenate([r[1] for r in results])
    all_qe  = np.concatenate([r[2] for r in results])
    all_dq  = np.concatenate([r[3] for r in results])
    all_dqe = np.concatenate([r[4] for r in results])
    all_lab = np.concatenate([r[5] for r in results])
    all_oa  = np.concatenate([r[6] for r in results])
    all_ob  = np.concatenate([r[7] for r in results])

    n_coll = int((all_lab < 0).sum())
    n_near = int(((all_lab >= 0) & (all_lab < 0.005)).sum())
    print()
    print(f"Stats:")
    print(f"  Total kept:              {len(all_lab)}")
    print(f"  Collisions (label<0):    {n_coll} ({100*n_coll/len(all_lab):.2f}%)")
    print(f"  Near-boundary (0<d<5mm): {n_near} ({100*n_near/len(all_lab):.2f}%)")
    print(f"  label  min/mean/max: "
          f"{all_lab.min():.4f} / {all_lab.mean():.4f} / {all_lab.max():.4f}")
    print(f"  d_q    min/mean/max: "
          f"{all_dq.min():.4f} / {all_dq.mean():.4f} / {all_dq.max():.4f}")
    print(f"  d_qe   min/mean/max: "
          f"{all_dqe.min():.4f} / {all_dqe.mean():.4f} / {all_dqe.max():.4f}")
    # how often does braking (qe) dominate the label?
    n_qe_binds = int((all_dqe < all_dq - 1e-9).sum())
    print(f"  qe binds label (d_qe<d_q): "
          f"{n_qe_binds} ({100*n_qe_binds/len(all_lab):.2f}%)")

    # Extended bins to expose DEEP penetration (no clip anywhere)
    bins = [-1.0, -0.30, -0.20, -0.10, -0.05, -0.02, -0.01, 0.0,
            0.005, 0.01, 0.02, 0.05, 0.1, 1.0]
    hist, _ = np.histogram(all_lab, bins=bins)
    print(f"\n  Label histogram (deep penetration kept):")
    for i in range(len(hist)):
        print(f"    [{bins[i]:+.4f}, {bins[i+1]:+.4f}): {hist[i]:>10}")

    # Quick offender-pair summary across boundary samples
    from collections import Counter
    bnd_mask = all_lab < 0.01
    pair_ctr = Counter()
    for a, b in zip(all_oa[bnd_mask], all_ob[bnd_mask]):
        if a and b:
            key = " <-> ".join(sorted([a, b]))
            pair_ctr[key] += 1
    print(f"\n  Top offender pairs (dist<10mm samples):")
    for key, cnt in pair_ctr.most_common(15):
        print(f"    x{cnt:>6}  {key}")
    n_bnd = int(bnd_mask.sum())
    n_inter = sum(c for k, c in pair_ctr.items()
                  if ("left" in k and "right" in k))
    print(f"\n  Inter-arm pairs in boundary samples: "
          f"{n_inter}/{n_bnd} ({100*n_inter/max(n_bnd,1):.1f}%)")

    out_path = os.path.join(args.output_dir, args.output_name)
    # Schema matches v2 CSV so train script can concat. final_pos = real qe
    # (braking stop pose), dist_qe = real d(qe), min_dist = viability label
    # = min(d_q, d_qe).  NO clipping anywhere.
    header = ([f"joint_{i}_pos" for i in range(14)]
              + [f"joint_{i}_vel" for i in range(14)]
              + [f"joint_{i}_final_pos" for i in range(14)]
              + ["dist_q", "dist_qe", "min_dist", "offender_a", "offender_b"])
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(all_q)):
            w.writerow(list(all_q[i]) + list(all_v[i]) + list(all_qe[i])
                       + [all_dq[i], all_dqe[i], all_lab[i],
                          all_oa[i], all_ob[i]])
    print(f"\n  Saved to: {out_path}")


if __name__ == "__main__":
    main()
