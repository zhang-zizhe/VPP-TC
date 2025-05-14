#!/usr/bin/env python3
import pybullet as p
import pybullet_data
import numpy as np
import csv
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(
        description='Sample joint configurations and detect self-collisions on a Panda robot')
    parser.add_argument(
        '--limit-sampling', action='store_true',
        help='Enable near-limit sampling for a subset of joints')
    parser.add_argument(
        '--limit-fraction', type=float, default=0.05,
        help='Fraction of joint range to sample near limits')
    parser.add_argument(
        '--limit-joints', type=int, default=3,
        help='Number of joints to sample near their limits')
    parser.add_argument(
        '--n-samples', type=int, default=100000,
        help='Total number of samples to generate')
    parser.add_argument(
        '--urdf-path', type=str,
        default=os.path.join(os.path.dirname(__file__), 'panda/panda.urdf'),
        help='Path to the Panda URDF file')
    return parser.parse_args()


def sample_configuration(joint_limits, args):
    """
    If limit_sampling is not enabled, perform uniform random sampling across all joints;
    otherwise, perform a global random sample, then randomly select args.limit_joints joints
    and sample near their lower or upper limits by args.limit_fraction.
    """
    # Base random sampling for all joints
    q = [np.random.uniform(low, high) for (low, high) in joint_limits]
    if not args.limit_sampling:
        return q

    # Resample near limits for a subset of joints
    num = min(args.limit_joints, len(joint_limits))
    idxs = np.random.choice(len(joint_limits), num, replace=False)
    for j in idxs:
        low, high = joint_limits[j]
        span = high - low
        frac = args.limit_fraction
        if np.random.rand() < 0.5:
            q[j] = np.random.uniform(low, low + frac * span)
        else:
            q[j] = np.random.uniform(high - frac * span, high)
    return q


def main():
    args = parse_args()

    # Connect to the simulation
    client = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    # Load the URDF with self-collision flags enabled
    flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
    robot = p.loadURDF(
        args.urdf_path,
        useFixedBase=True,
        flags=flags
    )
    # 获取关节索引和关节限制
    joint_indices = []
    joint_position_limits = []
    joint_velocity_limits = []
    joint_acceleration_limits = [(-15, 15), (-7.5, 7.5), (-10, 10), (-12.5, 12.5), (-15, 15), (-20, 20), (-20, 20)]
    joint_torque_limits = []
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        jtype = info[2]
        if jtype in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            joint_indices.append(i)
            joint_position_limits.append((info[8], info[9]))
            joint_velocity_limits.append((-info[11], info[11]))
            joint_torque_limits.append((-info[10], info[10]))
    # print("joint_position_limits", joint_position_limits)
    # print("joint_velocity_limits", joint_velocity_limits)
    # print("joint_acceleration_limits", joint_acceleration_limits)
    # print("joint_torque_limits", joint_torque_limits)
    if not joint_indices:
        raise RuntimeError(f'No movable joints found in URDF at {args.urdf_path}')

    # Generate samples and detect collisions
    n_samples = args.n_samples
    samples = []
    collision_flags = []
    N_collision = 0

    for idx in range(n_samples):
        q = sample_configuration(joint_position_limits, args)
        samples.append(q)

        # Apply joint states
        for jid, angle in zip(joint_indices, q):
            p.resetJointState(robot, jid, angle)
        p.stepSimulation()

        # Check for collisions
        contacts = p.getContactPoints(bodyA=robot, bodyB=robot)
        collision = len(contacts) > 0
        if collision:
            N_collision += 1
        collision_flags.append(collision)

        # Periodically print progress
        if (idx + 1) % (n_samples // 10) == 0:
            print(f"Sample {idx+1}/{n_samples}: collisions so far = {N_collision}")

    # Write results to CSV
    out_file = 'collision_results.csv'
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"joint_{i}" for i in range(len(joint_indices))] + ['collision']
        writer.writerow(header)
        for q, flag in zip(samples, collision_flags):
            writer.writerow(list(q) + [int(flag)])

    print(f"Done. Results saved to {out_file}")
    p.disconnect()


if __name__ == '__main__':
    main()
