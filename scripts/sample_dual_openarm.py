#!/usr/bin/env python3
"""Sample joint configurations and detect collisions on dual-arm OpenArm.

Generates a CSV dataset of (q, qd, qe, label) tuples where:
- q:     14 joint positions (7 per arm, randomly sampled — arm joints only)
- qd:    14 joint velocities (7 per arm, randomly sampled)
- qe:    14 predicted stopping positions (computed via max deceleration)
- label: 1 if either q or qe is in collision, 0 otherwise

Collisions include self-collision within each arm, inter-arm collisions,
AND arm–hand/finger collisions (the EE meshes participate in collision).

Supports parallel sampling via multiprocessing (one PyBullet DIRECT instance
per worker).

Usage
-----
    python scripts/sample_dual_openarm.py --n-samples 100000
    python scripts/sample_dual_openarm.py --n-samples 5000000 --workers 16
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
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, os.pardir)
sys.path.insert(0, os.path.abspath(_PROJECT_ROOT))

from vpptc.utils_openarm import DUAL_ACCELERATION_LIMITS
from vpptc.utils import compute_qe


def get_args():
    parser = argparse.ArgumentParser(
        description="Sample joint configurations and detect collisions (OpenArm dual-arm)",
    )
    parser.add_argument("--n-samples", type=int, default=10000000)
    parser.add_argument(
        "--urdf", type=str,
        default=os.path.join(
            _PROJECT_ROOT, "assets", "urdf", "openarm_description",
            "urdf", "robot", "openarm_bimanual.urdf"),
    )
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(_PROJECT_ROOT, "output"))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0 = nproc)")
    return parser.parse_args()


# ======================================================================
# Worker function — runs in a separate process
# ======================================================================

def _worker_init(urdf_path):
    """Initialise a per-process PyBullet DIRECT instance."""
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
    robot = p.loadURDF(urdf_path, useFixedBase=True, flags=flags,
                       physicsClientId=cid)

    # Collision filter: link5↔link7
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

    # Fix fingers
    finger_joints = []
    for i in range(p.getNumJoints(robot, physicsClientId=cid)):
        if "finger_joint" in p.getJointInfo(robot, i, physicsClientId=cid)[1].decode():
            finger_joints.append(i)
            p.resetJointState(robot, i, 0.01, physicsClientId=cid)

    # Discover arm joints
    arm_re = re.compile(r"openarm_(left|right)_joint[1-7]$")
    arm_joints = []
    arm_position_limits = []
    arm_velocity_limits = []
    for i in range(p.getNumJoints(robot, physicsClientId=cid)):
        info = p.getJointInfo(robot, i, physicsClientId=cid)
        jname = info[1].decode()
        if info[2] == p.JOINT_REVOLUTE and arm_re.match(jname):
            arm_joints.append(i)
            arm_position_limits.append((info[8], info[9]))
            arm_velocity_limits.append((-info[11], info[11]))

    return cid, robot, arm_joints, finger_joints, arm_position_limits, arm_velocity_limits


def _worker_fn(args_tuple):
    """Sample a chunk of configurations in one worker."""
    chunk_size, urdf_path, seed = args_tuple

    cid, robot, arm_joints, finger_joints, pos_lims, vel_lims = _worker_init(urdf_path)
    rng = np.random.RandomState(seed)

    results = []
    for _ in range(chunk_size):
        q = [rng.uniform(lo, hi) for lo, hi in pos_lims]
        qd = [rng.uniform(lo, hi) for lo, hi in vel_lims]
        qe = compute_qe(q, qd, acc_limits=DUAL_ACCELERATION_LIMITS)

        # Check q
        for jid, angle in zip(arm_joints, q):
            p.resetJointState(robot, jid, angle, physicsClientId=cid)
        for fj in finger_joints:
            p.resetJointState(robot, fj, 0.01, physicsClientId=cid)
        p.stepSimulation(physicsClientId=cid)
        col_q = len(p.getContactPoints(bodyA=robot, bodyB=robot,
                                       physicsClientId=cid)) > 0

        # Check qe
        for jid, angle in zip(arm_joints, qe):
            p.resetJointState(robot, jid, angle, physicsClientId=cid)
        for fj in finger_joints:
            p.resetJointState(robot, fj, 0.01, physicsClientId=cid)
        p.stepSimulation(physicsClientId=cid)
        col_qe = len(p.getContactPoints(bodyA=robot, bodyB=robot,
                                        physicsClientId=cid)) > 0

        collision = int(col_q or col_qe)
        results.append((q, qd, qe, collision))

    p.disconnect(physicsClientId=cid)
    return results


# ======================================================================
# Serial fallback (single-process)
# ======================================================================

def _sample_serial(n_samples, urdf_path, seed):
    """Original single-process sampling."""
    cid, robot, arm_joints, finger_joints, pos_lims, vel_lims = _worker_init(urdf_path)
    if seed is not None:
        np.random.seed(seed)

    results = []
    n_collision = 0
    report_interval = max(1, n_samples // 10)

    for idx in range(n_samples):
        q = [np.random.uniform(lo, hi) for lo, hi in pos_lims]
        qd = [np.random.uniform(lo, hi) for lo, hi in vel_lims]
        qe = compute_qe(q, qd, acc_limits=DUAL_ACCELERATION_LIMITS)

        for jid, angle in zip(arm_joints, q):
            p.resetJointState(robot, jid, angle, physicsClientId=cid)
        for fj in finger_joints:
            p.resetJointState(robot, fj, 0.01, physicsClientId=cid)
        p.stepSimulation(physicsClientId=cid)
        col_q = len(p.getContactPoints(bodyA=robot, bodyB=robot,
                                       physicsClientId=cid)) > 0

        for jid, angle in zip(arm_joints, qe):
            p.resetJointState(robot, jid, angle, physicsClientId=cid)
        for fj in finger_joints:
            p.resetJointState(robot, fj, 0.01, physicsClientId=cid)
        p.stepSimulation(physicsClientId=cid)
        col_qe = len(p.getContactPoints(bodyA=robot, bodyB=robot,
                                        physicsClientId=cid)) > 0

        collision = int(col_q or col_qe)
        if collision:
            n_collision += 1
        results.append((q, qd, qe, collision))

        if (idx + 1) % report_interval == 0:
            pct = n_collision / (idx + 1) * 100
            print(f"  [{idx + 1:>{len(str(n_samples))}}/{n_samples}] "
                  f"collisions = {n_collision} ({pct:.1f}%)")

    p.disconnect(physicsClientId=cid)
    return results


# ======================================================================
# Main
# ======================================================================

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    n_workers = args.workers if args.workers > 0 else mp.cpu_count()
    base_seed = args.seed if args.seed is not None else 42

    print(f"Sampling {args.n_samples} configurations with {n_workers} workers ...")
    t0 = time.time()

    if n_workers == 1:
        results = _sample_serial(args.n_samples, args.urdf, base_seed)
    else:
        # Split work into chunks
        chunk_size = args.n_samples // n_workers
        remainder = args.n_samples % n_workers
        tasks = []
        for w in range(n_workers):
            n = chunk_size + (1 if w < remainder else 0)
            tasks.append((n, args.urdf, base_seed + w))

        with mp.Pool(n_workers) as pool:
            chunk_results = pool.map(_worker_fn, tasks)

        results = []
        for chunk in chunk_results:
            results.extend(chunk)

    elapsed = time.time() - t0
    n_collision = sum(r[3] for r in results)
    total_pct = n_collision / len(results) * 100
    rate = len(results) / elapsed

    print(f"\nDone in {elapsed:.1f}s ({rate:.0f} samples/s)")
    print(f"  {n_collision}/{len(results)} collisions ({total_pct:.1f}%)")

    # --- Write CSV ---
    n_joints = 14
    csv_path = os.path.join(args.output_dir, "openarm_dual_collision_results.csv")
    header = (
        [f"joint_{i}_pos" for i in range(n_joints)]
        + [f"joint_{i}_vel" for i in range(n_joints)]
        + [f"joint_{i}_final_pos" for i in range(n_joints)]
        + ["collision"]
    )
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for q, qd, qe, flag in results:
            writer.writerow(list(q) + list(qd) + list(qe) + [flag])

    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
