#!/usr/bin/env python3
"""Sample joint configurations and detect self-collisions on a Panda robot.

Generates a CSV dataset of (q, qd, qe, label) tuples where:
- q:     7 joint positions (randomly sampled)
- qd:    7 joint velocities (randomly sampled)
- qe:    7 predicted stopping positions (computed via max deceleration)
- label: 1 if either q or qe is in self-collision, 0 otherwise

The output CSV is directly consumable by ``scripts/train.py``.

Usage
-----
    python scripts/sample.py --n-samples 100000
    python scripts/sample.py --n-samples 200000 --limit-sampling --limit-joints 3
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


# ======================================================================
# Argument parsing
# ======================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="Sample joint configurations and detect self-collisions",
    )
    parser.add_argument(
        "--n-samples", type=int, default=100000,
        help="Total number of samples to generate (default: 100000)",
    )
    parser.add_argument(
        "--limit-sampling", action="store_true",
        help="Enable near-limit sampling for a subset of joints",
    )
    parser.add_argument(
        "--limit-fraction", type=float, default=0.05,
        help="Fraction of joint range to sample near limits (default: 0.05)",
    )
    parser.add_argument(
        "--limit-joints", type=int, default=3,
        help="Number of joints to sample near their limits (default: 3)",
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

def sample_configuration(joint_position_limits, joint_velocity_limits, args):
    """Sample a random (q, qd) pair.

    If ``--limit-sampling`` is enabled, a subset of joints are re-sampled
    near their lower or upper position limits (within *limit_fraction* of
    the total range).  This biases the dataset toward boundary regions
    where self-collisions are more likely.
    """
    q = [np.random.uniform(lo, hi) for (lo, hi) in joint_position_limits]
    qd = [np.random.uniform(lo, hi) for (lo, hi) in joint_velocity_limits]

    if not args.limit_sampling:
        return q, qd

    # Re-sample near limits for a random subset of joints
    n_joints = len(joint_position_limits)
    num = min(args.limit_joints, n_joints)
    idxs = np.random.choice(n_joints, num, replace=False)
    for j in idxs:
        lo, hi = joint_position_limits[j]
        span = hi - lo
        frac = args.limit_fraction
        if np.random.rand() < 0.5:
            q[j] = np.random.uniform(lo, lo + frac * span)
        else:
            q[j] = np.random.uniform(hi - frac * span, hi)

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

    if not joint_indices:
        raise RuntimeError(f"No movable joints found in URDF at {args.urdf}")

    n_joints = len(joint_indices)
    print(f"Found {n_joints} movable joints")
    print(f"Sampling {args.n_samples} configurations "
          f"(limit_sampling={'ON' if args.limit_sampling else 'OFF'}) ...")

    # --- Sample and check collisions ---
    pos_samples = []
    vel_samples = []
    end_samples = []
    collision_flags = []
    n_collision = 0

    report_interval = max(1, args.n_samples // 10)

    for idx in range(args.n_samples):
        q, qd = sample_configuration(
            joint_position_limits, joint_velocity_limits, args)

        # Predicted stopping position using max deceleration
        qe = compute_qe(q, qd)

        # Check self-collision at current pose q
        collision_q = check_self_collision(robot, joint_indices, q)

        # Check self-collision at stopping pose qe
        collision_qe = check_self_collision(robot, joint_indices, qe)

        # Mark as collision if either pose collides
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
    if args.limit_sampling:
        filename = f"collision_results_{args.limit_joints}_limit_sampling.csv"
    else:
        filename = "collision_results.csv"
    csv_path = os.path.join(args.output_dir, filename)

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
