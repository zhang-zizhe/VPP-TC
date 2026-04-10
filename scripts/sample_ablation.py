#!/usr/bin/env python3
"""Ablation sampling: position-only collision label + random velocity (single-arm).

Unlike the standard sampler which uses viability (qe via max deceleration),
this script labels each sample based on the current position only:
- collision at q → label = 1
- no collision at q → label = 0

Velocities are independently random and do NOT affect the label.
The qe column is still included (filled with random values) to keep
the CSV format identical for downstream training.

Usage
-----
    python scripts/sample_ablation.py --n-samples 10000000
"""

import argparse
import csv
import os
import sys

import numpy as np
import pybullet as p
import pybullet_data

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, os.pardir)
sys.path.insert(0, os.path.abspath(_PROJECT_ROOT))


# ======================================================================
# Argument parsing
# ======================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Ablation: position-only collision sampling (single-arm)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=3000000,
        help="Total number of samples to generate (default: 10000000)",
    )
    parser.add_argument(
        "--urdf", type=str,
        default=os.path.join(_PROJECT_ROOT, "assets", "urdf", "panda", "panda.urdf"),
        help="Path to the Panda URDF file",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=os.path.join(_PROJECT_ROOT, "output"),
        help="Directory for the output CSV (default: output/)",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


# ======================================================================
# Sampling helpers
# ======================================================================

def sample_configuration(joint_position_limits, joint_velocity_limits):
    """Sample a random (q, qd) pair for 7 joints."""
    q = [np.random.uniform(lo, hi) for (lo, hi) in joint_position_limits]
    qd = [np.random.uniform(lo, hi) for (lo, hi) in joint_velocity_limits]
    return q, qd


def check_self_collision(robot_id, joint_indices, q):
    """Set the robot to configuration *q* and return True if self-collision."""
    for jid, angle in zip(joint_indices, q):
        p.resetJointState(robot_id, jid, angle)
    p.stepSimulation()
    contacts = p.getContactPoints(bodyA=robot_id, bodyB=robot_id)
    return len(contacts) > 0


# ======================================================================
# Main
# ======================================================================

def main():
    args = get_args()
    if args.seed is not None:
        np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # --- PyBullet setup (headless) ---
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
    robot = p.loadURDF(args.urdf, useFixedBase=True, flags=flags)

    # Exclude link pair (4, 6) from self-collision (consistent with simulate.py)
    p.setCollisionFilterPair(robot, robot, 4, 6, enableCollision=0)

    # --- Discover movable joints and their limits ---
    joint_indices = []
    joint_position_limits = []
    joint_velocity_limits = []
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            joint_indices.append(i)
            joint_position_limits.append((info[8], info[9]))
            joint_velocity_limits.append((-info[11], info[11]))

    n_joints = len(joint_indices)
    print(f"Found {n_joints} movable joints")
    print(f"[ABLATION] Position-only labelling (no viability qe)")
    print(f"Sampling {args.n_samples} configurations ...")

    # --- Sample with balanced ratio (collision:safe = 4:6) ---
    max_collision = int(args.n_samples * 0.4)
    max_safe = args.n_samples - max_collision
    pos_samples = []
    vel_samples = []
    collision_flags = []
    n_collision = 0
    n_safe = 0
    n_tried = 0

    report_interval = max(1, args.n_samples // 10)

    while (n_collision + n_safe) < args.n_samples:
        q, qd = sample_configuration(
            joint_position_limits, joint_velocity_limits)
        n_tried += 1

        # Ablation: label based on position q only, no viability
        collision = check_self_collision(robot, joint_indices, q)

        # Reject if quota for this class is full
        if collision and n_collision >= max_collision:
            continue
        if not collision and n_safe >= max_safe:
            continue

        if collision:
            n_collision += 1
        else:
            n_safe += 1

        pos_samples.append(q)
        vel_samples.append(qd)
        collision_flags.append(collision)

        total = n_collision + n_safe
        if total % report_interval == 0:
            pct = n_collision / total * 100
            print(f"  [{total:>{len(str(args.n_samples))}}/{args.n_samples}] "
                  f"collisions = {n_collision} ({pct:.1f}%)  "
                  f"tried = {n_tried}")

    # --- Write CSV ---
    csv_path = os.path.join(args.output_dir, "ablation_posonly_3M.csv")

    header = (
        [f"joint_{i}_pos" for i in range(n_joints)]
        + [f"joint_{i}_vel" for i in range(n_joints)]
        + ["collision"]
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for q, qd, flag in zip(
            pos_samples, vel_samples, collision_flags
        ):
            writer.writerow(list(q) + list(qd) + [int(flag)])

    total_pct = n_collision / args.n_samples * 100
    print(f"\nDone. {n_collision}/{args.n_samples} collisions ({total_pct:.1f}%)")
    print(f"Results saved to {csv_path}")
    p.disconnect()


if __name__ == "__main__":
    main()
