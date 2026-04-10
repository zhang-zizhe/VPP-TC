#!/usr/bin/env python3
"""Sample joint configurations and detect collisions on dual-arm Panda.

Generates a CSV dataset of (q, qd, qe, label) tuples where:
- q:     14 joint positions (7 per arm, randomly sampled)
- qd:    14 joint velocities (7 per arm, randomly sampled)
- qe:    14 predicted stopping positions (computed via max deceleration)
- label: 1 if either q or qe is in collision, 0 otherwise

Collisions include both self-collision within each arm **and** inter-arm
collisions.

Usage
-----
    python scripts/sample_dual.py --n-samples 100000
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

from vpptc.utils import JOINT_ACCELERATION_LIMITS, compute_qe

DUAL_ACCELERATION_LIMITS = JOINT_ACCELERATION_LIMITS + JOINT_ACCELERATION_LIMITS


# ======================================================================
# Argument parsing
# ======================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Sample joint configurations and detect collisions (dual-arm)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=10000000,
        help="Total number of samples to generate (default: 10000000)",
    )
    parser.add_argument(
        "--urdf", type=str,
        default=os.path.join(
            _PROJECT_ROOT, "assets", "urdf", "panda", "panda_dual_arms.urdf"),
        help="Path to the dual-arm URDF file",
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
    """Sample a random (q, qd) pair for 14 joints (two 7-DOF arms)."""
    q = [np.random.uniform(lo, hi) for (lo, hi) in joint_position_limits]
    qd = [np.random.uniform(lo, hi) for (lo, hi) in joint_velocity_limits]
    return q, qd


def check_collision(robot_id, joint_indices, q):
    """Set the robot to configuration *q* and return True if any collision.

    This catches both self-collision within each arm and inter-arm collisions.
    """
    for jid, angle in zip(joint_indices, q):
        p.resetJointState(robot_id, jid, angle)
    p.stepSimulation()
    contacts = p.getContactPoints(bodyA=robot_id, bodyB=robot_id)
    return len(contacts) > 0


def find_collision_filter_pairs(robot_id):
    """Find (link5, link7) pairs for each arm to exclude from collision.

    Returns a list of (linkA_index, linkB_index) tuples.
    """
    link5_indices = []
    link7_indices = []
    for i in range(p.getNumJoints(robot_id)):
        name = p.getJointInfo(robot_id, i)[12].decode("utf-8")
        if name.endswith("_link5"):
            link5_indices.append(i)
        elif name.endswith("_link7"):
            link7_indices.append(i)

    pairs = []
    for l5 in link5_indices:
        for l7 in link7_indices:
            if abs(l5 - l7) < 5:
                pairs.append((l5, l7))
    return pairs


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

    # Exclude (link5, link7) pairs for each arm (consistent with simulate.py)
    filter_pairs = find_collision_filter_pairs(robot)
    for l_a, l_b in filter_pairs:
        p.setCollisionFilterPair(robot, robot, l_a, l_b, enableCollision=0)
        name_a = p.getJointInfo(robot, l_a)[12].decode("utf-8")
        name_b = p.getJointInfo(robot, l_b)[12].decode("utf-8")
        print(f"Excluded collision pair: ({l_a}) {name_a} <-> ({l_b}) {name_b}")

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
    if n_joints != 14:
        raise RuntimeError(
            f"Expected 14 movable joints for dual-arm, found {n_joints}")

    print(f"Found {n_joints} movable joints (2 arms x 7 DOF)")
    print(f"Sampling {args.n_samples} configurations ...")

    # --- Sample and check collisions ---
    pos_samples = []
    vel_samples = []
    end_samples = []
    collision_flags = []
    n_collision = 0

    report_interval = max(1, args.n_samples // 10)

    for idx in range(args.n_samples):
        q, qd = sample_configuration(
            joint_position_limits, joint_velocity_limits)

        qe = compute_qe(q, qd, acc_limits=DUAL_ACCELERATION_LIMITS)

        collision_q = check_collision(robot, joint_indices, q)
        collision_qe = check_collision(robot, joint_indices, qe)

        collision = collision_q or collision_qe
        if collision:
            n_collision += 1

        pos_samples.append(q)
        vel_samples.append(qd)
        end_samples.append(qe)
        collision_flags.append(collision)

        if (idx + 1) % report_interval == 0:
            pct = n_collision / (idx + 1) * 100
            print(f"  [{idx + 1:>{len(str(args.n_samples))}}/{args.n_samples}] "
                  f"collisions = {n_collision} ({pct:.1f}%)")

    # --- Write CSV ---
    csv_path = os.path.join(args.output_dir, "dual_collision_d0.6m.csv")

    header = (
        [f"joint_{i}_pos" for i in range(n_joints)]
        + [f"joint_{i}_vel" for i in range(n_joints)]
        + [f"joint_{i}_final_pos" for i in range(n_joints)]
        + ["collision"]
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for q, qd, qe, flag in zip(
            pos_samples, vel_samples, end_samples, collision_flags
        ):
            writer.writerow(list(q) + list(qd) + list(qe) + [int(flag)])

    total_pct = n_collision / args.n_samples * 100
    print(f"\nDone. {n_collision}/{args.n_samples} collisions ({total_pct:.1f}%)")
    print(f"Results saved to {csv_path}")
    p.disconnect()


if __name__ == "__main__":
    main()
