#!/usr/bin/env python3
"""Sample OpenArm dual-arm joint configs and record min self-collision DISTANCE.

Mirrors scripts/sample_dist.py (single-arm Panda) but for the OpenArm
bimanual robot.  Instead of a binary collision flag, records the minimum
closest-point distance over all (filter-enabled) link pairs, evaluated
at both q and the stopping pose qe.

Output CSV columns (43 total):
    joint_0_pos ... joint_13_pos       (14 joint positions)
    joint_0_vel ... joint_13_vel       (14 joint velocities)
    joint_0_final_pos ... _13_final_pos (14 stopping pose qe)
    dist_q                              (min closest-point dist at q)
    dist_qe                             (min closest-point dist at qe)
    min_dist                            (min of the two, clipped)

Negative values  => penetration (self-collision)
Positive values  => safe, value is the gap in metres
Capped at +max_check_dist (far-field flat) and at clip_min (deep penetration).

Usage
-----
    python scripts/sample_dual_openarm_dist.py --n-samples 3000000 --workers 8
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

from vpptc.utils_openarm import DUAL_ACCELERATION_LIMITS, DUAL_POS_LIMITS
from vpptc.utils import compute_qe


# ======================================================================
# Args
# ======================================================================

def get_args():
    ap = argparse.ArgumentParser(
        description="OpenArm dual-arm: sample (q,qd) and record min self-coll distance")
    ap.add_argument("--n-samples", type=int, default=3_000_000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-check-dist", type=float, default=0.2,
                    help="Max search distance for getClosestPoints (cap value)")
    ap.add_argument("--clip-min", type=float, default=-0.05,
                    help="Clip negative distance to this value (cap penetration)")
    ap.add_argument("--urdf", type=str,
                    default=os.path.join(
                        _PROJECT_ROOT, "assets", "urdf", "openarm_description",
                        "urdf", "robot", "openarm_bimanual.urdf"))
    ap.add_argument("--output-dir", type=str,
                    default=os.path.join(_PROJECT_ROOT, "output"))
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


# ======================================================================
# Per-process PyBullet setup (same filter scheme as sample_dual_openarm.py)
# ======================================================================

def _setup(urdf_path):
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
    robot = p.loadURDF(urdf_path, useFixedBase=True, flags=flags,
                       physicsClientId=cid)

    # Disable link5 <-> link7 (within each arm) — same as sample_dual_openarm.py
    link5s, link7s = [], []
    for i in range(p.getNumJoints(robot, physicsClientId=cid)):
        name = p.getJointInfo(robot, i, physicsClientId=cid)[12].decode()
        if name.endswith("_link5"):
            link5s.append(i)
        elif name.endswith("_link7"):
            link7s.append(i)
    for l5 in link5s:
        for l7 in link7s:
            if abs(l5 - l7) < 5:
                p.setCollisionFilterPair(robot, robot, l5, l7,
                                         enableCollision=0, physicsClientId=cid)

    # Fingers held at fixed open pose
    finger_joints = []
    for i in range(p.getNumJoints(robot, physicsClientId=cid)):
        if "finger_joint" in p.getJointInfo(robot, i, physicsClientId=cid)[1].decode():
            finger_joints.append(i)
            p.resetJointState(robot, i, 0.01, physicsClientId=cid)

    # Discover the 14 arm joints
    arm_re = re.compile(r"openarm_(left|right)_joint[1-7]$")
    arm_joints, pos_lims, vel_lims = [], [], []
    for i in range(p.getNumJoints(robot, physicsClientId=cid)):
        info = p.getJointInfo(robot, i, physicsClientId=cid)
        jname = info[1].decode()
        if info[2] == p.JOINT_REVOLUTE and arm_re.match(jname):
            arm_joints.append(i)
            pos_lims.append((info[8], info[9]))
            vel_lims.append((-info[11], info[11]))

    # ----- Blacklist: load auto-generated structural pair list -----
    # Pairs derived by scripts/auto_blacklist_openarm.py from a 5K-probe scan.
    # Combines (a) parent-child + same-arm link5<->link7 URDF artefacts and
    # (b) intra-arm pairs proven to never penetrate during probing.
    n_links = p.getNumJoints(robot, physicsClientId=cid)
    name_to_idx = {}
    for i in range(n_links):
        nm = p.getJointInfo(robot, i, physicsClientId=cid)[12].decode()
        name_to_idx[nm] = i

    from vpptc.blacklist_openarm import BLACKLIST_NAME_PAIRS
    blacklist = set()
    n_missing = 0
    for na, nb in BLACKLIST_NAME_PAIRS:
        if na in name_to_idx and nb in name_to_idx:
            ia, ib = name_to_idx[na], name_to_idx[nb]
            blacklist.add((min(ia, ib), max(ia, ib)))
        else:
            n_missing += 1
    # Inter-arm structural pairs: link0 is rigidly mounted to body and link1
    # is rotationally symmetric about joint1's axis, so all 4 pairwise
    # combinations of {link0,link1}_L x {link0,link1}_R have CONSTANT
    # inter-arm distance (~0.060 m / ~0.123 m) regardless of pose.  Without
    # excluding them, min_dist gets pinned at 0.060 m for ~93% of poses,
    # destroying the regression target's dynamic range.
    INTER_STRUCT_PAIRS = [
        ("openarm_left_link0", "openarm_right_link0"),
        ("openarm_left_link0", "openarm_right_link1"),
        ("openarm_left_link1", "openarm_right_link0"),
        ("openarm_left_link1", "openarm_right_link1"),
    ]
    for na, nb in INTER_STRUCT_PAIRS:
        if na in name_to_idx and nb in name_to_idx:
            ia, ib = name_to_idx[na], name_to_idx[nb]
            blacklist.add((min(ia, ib), max(ia, ib)))

    print(f"  [blacklist] loaded {len(blacklist)} pairs "
          f"(intra from vpptc/blacklist_openarm.py + 4 inter-arm structural; "
          f"{n_missing} intra skipped: link not found)",
          flush=True)

    return cid, robot, arm_joints, finger_joints, pos_lims, vel_lims, blacklist


def min_self_distance(cid, robot, arm_joints, finger_joints, q, max_d,
                       blacklist, debug_pairs=None):
    """Return min closest-point distance across non-blacklisted self pairs.

    If debug_pairs is a list, append (a, b, dist) of the offending pair.
    """
    for jid, angle in zip(arm_joints, q):
        p.resetJointState(robot, jid, angle, physicsClientId=cid)
    for fj in finger_joints:
        p.resetJointState(robot, fj, 0.01, physicsClientId=cid)
    p.stepSimulation(physicsClientId=cid)

    pts = p.getClosestPoints(bodyA=robot, bodyB=robot,
                             distance=max_d, physicsClientId=cid)
    if not pts:
        return max_d
    d = max_d
    worst = None
    for pt in pts:
        a, b = pt[3], pt[4]
        if a == b:
            continue
        if (min(a, b), max(a, b)) in blacklist:
            continue
        if pt[8] < d:
            d = pt[8]
            worst = (a, b, pt[8])
    if debug_pairs is not None and worst is not None:
        debug_pairs.append(worst)
    return d


# ======================================================================
# Worker
# ======================================================================

_shared_counter = None


def _worker_init(counter):
    global _shared_counter
    _shared_counter = counter


def _worker(task):
    n_target, seed, urdf_path, max_d, clip_min = task
    cid, robot, arm_joints, finger_joints, pos_lims, vel_lims, blacklist = _setup(urdf_path)
    rng = np.random.default_rng(seed)

    out_q, out_v, out_qe, out_dq, out_dqe, out_label = [], [], [], [], [], []
    local_acc = 0
    debug_pairs = []   # collect first 20 offenders for diagnostics
    link_name = {-1: "base"}
    for i in range(p.getNumJoints(robot, physicsClientId=cid)):
        link_name[i] = p.getJointInfo(robot, i, physicsClientId=cid)[12].decode()

    for _ in range(n_target):
        q = rng.uniform([lo for lo, _ in pos_lims], [hi for _, hi in pos_lims])
        v = rng.uniform([lo for lo, _ in vel_lims], [hi for _, hi in vel_lims])
        qe = compute_qe(q.tolist(), v.tolist(), acc_limits=DUAL_ACCELERATION_LIMITS,
                        pos_limits=DUAL_POS_LIMITS)

        dbg = debug_pairs if len(debug_pairs) < 20 else None
        d_q = min_self_distance(cid, robot, arm_joints, finger_joints, q, max_d, blacklist, dbg)
        d_qe = min_self_distance(cid, robot, arm_joints, finger_joints, qe, max_d, blacklist, dbg)
        d_min = min(d_q, d_qe)
        if d_min < clip_min:
            d_min = clip_min

        out_q.append(q); out_v.append(v); out_qe.append(np.asarray(qe))
        out_dq.append(d_q); out_dqe.append(d_qe); out_label.append(d_min)
        local_acc += 1

        if local_acc % 500 == 0 and _shared_counter is not None:
            with _shared_counter.get_lock():
                _shared_counter.value += 500

    if local_acc % 500 != 0 and _shared_counter is not None:
        with _shared_counter.get_lock():
            _shared_counter.value += local_acc % 500

    # One-shot diagnostic: which link pairs produced the minimum distance?
    if debug_pairs:
        from collections import Counter
        ctr = Counter()
        for a, b, dist in debug_pairs:
            ctr[(min(a, b), max(a, b))] += 1
        print(f"  [worker seed={seed}] first {len(debug_pairs)} offenders, "
              f"top pairs causing min_dist:", flush=True)
        for (a, b), n in ctr.most_common(8):
            sample_dist = next(d for aa, bb, d in debug_pairs
                                if (min(aa, bb), max(aa, bb)) == (a, b))
            print(f"    x{n:3d}  {link_name.get(a, a):35s} <-> "
                  f"{link_name.get(b, b):35s}  e.g. dist={sample_dist:+.4f}",
                  flush=True)

    p.disconnect(physicsClientId=cid)
    return (np.array(out_q), np.array(out_v), np.array(out_qe),
            np.array(out_dq), np.array(out_dqe), np.array(out_label))


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
                print(f"  [{min(n, n_target):>{width}}/{n_target}]  "
                      f"elapsed={time.time() - t0:.1f}s")
                next_report = (n // report_every + 1) * report_every
        results = ar.get()

    print(f"  Done in {time.time() - t0:.1f}s")
    return results


# ======================================================================
# Main
# ======================================================================

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    n_workers = max(1, min(args.workers, args.n_samples))
    print(f"OpenArm dual-arm distance sampling")
    print(f"  Total samples:      {args.n_samples}")
    print(f"  Workers:            {n_workers}")
    print(f"  Max check distance: {args.max_check_dist} m")
    print(f"  Clip min:           {args.clip_min} m")
    print()

    base, rem = divmod(args.n_samples, n_workers)
    counts = [base + (1 if i < rem else 0) for i in range(n_workers)]
    rng_root = np.random.SeedSequence(args.seed)
    seeds = rng_root.spawn(n_workers)
    tasks = [(counts[i], seeds[i], args.urdf,
              args.max_check_dist, args.clip_min)
             for i in range(n_workers)]

    results = run_parallel(tasks, args.n_samples, n_workers)

    all_q   = np.concatenate([r[0] for r in results])
    all_v   = np.concatenate([r[1] for r in results])
    all_qe  = np.concatenate([r[2] for r in results])
    all_dq  = np.concatenate([r[3] for r in results])
    all_dqe = np.concatenate([r[4] for r in results])
    all_lab = np.concatenate([r[5] for r in results])

    n_coll = int((all_lab < 0).sum())
    n_near = int(((all_lab >= 0) & (all_lab < 0.005)).sum())
    print()
    print(f"Stats:")
    print(f"  Collisions (dist<0):     {n_coll} ({100*n_coll/len(all_lab):.2f}%)")
    print(f"  Near-boundary (0<d<5mm): {n_near} ({100*n_near/len(all_lab):.2f}%)")
    print(f"  min/mean/max label: {all_lab.min():.4f} / {all_lab.mean():.4f} / {all_lab.max():.4f}")

    bins = [-0.05, -0.01, 0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 1.0]
    hist, _ = np.histogram(all_lab, bins=bins)
    print(f"\n  Distance histogram:")
    for i in range(len(hist)):
        print(f"    [{bins[i]:+.4f}, {bins[i+1]:+.4f}): {hist[i]:>10}")

    n_joints = 14
    out_path = os.path.join(args.output_dir, "openarm_dual_collision_distance.csv")
    header = ([f"joint_{i}_pos" for i in range(n_joints)]
              + [f"joint_{i}_vel" for i in range(n_joints)]
              + [f"joint_{i}_final_pos" for i in range(n_joints)]
              + ["dist_q", "dist_qe", "min_dist"])

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for i in range(len(all_q)):
            w.writerow(list(all_q[i]) + list(all_v[i]) + list(all_qe[i])
                       + [all_dq[i], all_dqe[i], all_lab[i]])

    print(f"\n  Saved to: {out_path}")


if __name__ == "__main__":
    main()
