#!/usr/bin/env python3
"""Diagnose: in samples where min_dist<10mm, which link pairs are the offenders?

Subsample rows from the v2 CSV that have small min_dist, re-run PyBullet
getClosestPoints, and tally the closest pair by link-name.
"""
import os, sys, re, argparse
from collections import Counter

import numpy as np
import pandas as pd
import pybullet as p
import pybullet_data

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
sys.path.insert(0, _ROOT)

from vpptc.blacklist_openarm import BLACKLIST_NAME_PAIRS


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="output/openarm_dual_collision_distance_v2.csv")
    ap.add_argument("--urdf", default="assets/urdf/openarm_description/urdf/robot/openarm_bimanual.urdf")
    ap.add_argument("--threshold", type=float, default=0.01,
                    help="Only diagnose samples with min_dist < this (m)")
    ap.add_argument("--n", type=int, default=3000,
                    help="How many sub-threshold samples to diagnose")
    ap.add_argument("--chunk-rows", type=int, default=2_000_000,
                    help="Read CSV in chunks of this many rows")
    return ap.parse_args()


def setup(urdf_path):
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
    robot = p.loadURDF(urdf_path, useFixedBase=True, flags=flags, physicsClientId=cid)

    link5s, link7s = [], []
    name_to_idx, idx_to_name = {}, {}
    finger_joints, arm_joints = [], []
    arm_re = re.compile(r"openarm_(left|right)_joint[1-7]$")
    for i in range(p.getNumJoints(robot, physicsClientId=cid)):
        info = p.getJointInfo(robot, i, physicsClientId=cid)
        jname = info[1].decode()
        lname = info[12].decode()
        name_to_idx[lname] = i
        idx_to_name[i] = lname
        if lname.endswith("_link5"): link5s.append(i)
        if lname.endswith("_link7"): link7s.append(i)
        if "finger_joint" in jname:
            finger_joints.append(i)
            p.resetJointState(robot, i, 0.01, physicsClientId=cid)
        if info[2] == p.JOINT_REVOLUTE and arm_re.match(jname):
            arm_joints.append(i)

    for l5 in link5s:
        for l7 in link7s:
            if abs(l5 - l7) < 5:
                p.setCollisionFilterPair(robot, robot, l5, l7,
                                         enableCollision=0, physicsClientId=cid)

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

    return cid, robot, arm_joints, finger_joints, idx_to_name, blacklist


def closest_pair(cid, robot, arm_joints, finger_joints, q, max_d, blacklist):
    for jid, angle in zip(arm_joints, q):
        p.resetJointState(robot, jid, angle, physicsClientId=cid)
    for fj in finger_joints:
        p.resetJointState(robot, fj, 0.01, physicsClientId=cid)
    p.stepSimulation(physicsClientId=cid)
    pts = p.getClosestPoints(bodyA=robot, bodyB=robot,
                             distance=max_d, physicsClientId=cid)
    best = None
    for pt in pts:
        a, b = pt[3], pt[4]
        if a == b: continue
        if (min(a,b), max(a,b)) in blacklist: continue
        if best is None or pt[8] < best[2]:
            best = (a, b, pt[8])
    return best


def simplify(name):
    """openarm_left_link4 -> L_link4, openarm_right_right_finger -> R_rfinger"""
    s = name.replace("openarm_left_", "L_").replace("openarm_right_", "R_")
    s = s.replace("right_finger", "rfinger").replace("left_finger", "lfinger")
    return s


def main():
    args = get_args()
    urdf = os.path.join(_ROOT, args.urdf) if not os.path.isabs(args.urdf) else args.urdf
    csv = os.path.join(_ROOT, args.csv) if not os.path.isabs(args.csv) else args.csv

    print(f"Streaming {csv} for rows with min_dist < {args.threshold*1000:.0f}mm "
          f"(target n={args.n}) ...")
    pos_cols = [f"joint_{i}_pos" for i in range(14)]
    collected = []
    total_rows = 0
    sub_rows = 0
    for chunk in pd.read_csv(csv, usecols=pos_cols + ["min_dist"],
                             chunksize=args.chunk_rows):
        total_rows += len(chunk)
        sub = chunk[chunk["min_dist"] < args.threshold]
        sub_rows += len(sub)
        if len(collected) < args.n:
            need = args.n - len(collected)
            collected.append(sub.head(need))
            if sum(len(c) for c in collected) >= args.n:
                # but we still want overall counts -> keep reading
                pass
        if total_rows >= 10_000_000:
            break
    df = pd.concat(collected, ignore_index=True) if collected else pd.DataFrame()
    print(f"  total rows scanned : {total_rows:,}")
    print(f"  rows < {args.threshold*1000:.0f}mm    : {sub_rows:,} "
          f"({sub_rows/total_rows*100:.2f}%)")
    print(f"  diagnosing         : {len(df):,}")
    print()

    cid, robot, arm_joints, finger_joints, idx_to_name, blacklist = setup(urdf)

    pair_counter = Counter()
    finger_involved = 0
    finger_finger = 0
    pair_min_dist = {}  # pair -> list of dists for stats
    for i, row in df.iterrows():
        q = row[pos_cols].values.astype(np.float64)
        best = closest_pair(cid, robot, arm_joints, finger_joints, q, 0.05, blacklist)
        if best is None:
            continue
        a, b, d = best
        na, nb = simplify(idx_to_name[a]), simplify(idx_to_name[b])
        key = " <-> ".join(sorted([na, nb]))
        pair_counter[key] += 1
        pair_min_dist.setdefault(key, []).append(d)
        if "finger" in key:
            finger_involved += 1
        if key.count("finger") == 2:
            finger_finger += 1

    n_diag = sum(pair_counter.values())
    print("=" * 78)
    print(f"OFFENDING-PAIR HISTOGRAM (top 20, n={n_diag} diagnosed)")
    print(f"  {'pair':<55} {'count':>7} {'%':>6} {'med d (mm)':>11}")
    for key, cnt in pair_counter.most_common(20):
        med = np.median(pair_min_dist[key]) * 1000
        print(f"  {key:<55} {cnt:>7d} {cnt/n_diag*100:>5.1f}% {med:>10.2f}")
    print()
    print(f"finger-involved pairs : {finger_involved} ({finger_involved/n_diag*100:.1f}%)")
    print(f"  of which finger-finger : {finger_finger} ({finger_finger/n_diag*100:.1f}%)")
    print(f"  arm-finger             : {finger_involved-finger_finger} "
          f"({(finger_involved-finger_finger)/n_diag*100:.1f}%)")
    print(f"arm-arm only          : {n_diag-finger_involved} "
          f"({(n_diag-finger_involved)/n_diag*100:.1f}%)")


if __name__ == "__main__":
    main()
