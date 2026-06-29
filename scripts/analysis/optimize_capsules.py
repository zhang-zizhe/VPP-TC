#!/usr/bin/env python3
"""Auto-diagnose and optimize per-link capsule parameters.

Two-phase analysis:

Phase 1 - Coverage check: for each link, verify the current capsule
fully encloses its collision mesh.  If not, report the minimum radius
required (= max perpendicular distance from any mesh vertex to the
capsule axis line).  Under-covered capsules risk false negatives
(say "safe" when mesh actually collides).

Phase 2 - FP attribution: sample random poses, compare mesh vs capsule
distance, and for poses where mesh is safe (>2cm) but capsule reports
collision, find which capsule PAIR is the worst offender.  These pairs
suggest one of the two capsules is too fat or its axis is wrong.

Output: a sorted list of per-link recommendations, plus the worst
FP-causing pairs to inspect/blacklist.
"""
import os, sys, re
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import trimesh
import pybullet as p
import pybullet_data

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
sys.path.insert(0, _ROOT)

from vpptc.capsules_openarm import all_capsule_definitions, build_blacklist_pairs
from vpptc.capsule_distance import (segment_segment_distance_np,
                                      capsule_capsule_distance_np,
                                      transform_capsule)

URDF = os.path.join(_ROOT, "assets", "urdf", "openarm_description",
                    "urdf", "robot", "openarm_bimanual.urdf")
URDF_DIR = os.path.dirname(URDF)


# ----------------------------------------------------------------------
# Phase 1: coverage check per link
# ----------------------------------------------------------------------

def parse_origin(elem):
    o = elem.find("origin")
    if o is None:
        return np.zeros(3), np.zeros(3)
    xyz = np.array([float(x) for x in o.get("xyz", "0 0 0").split()])
    rpy = np.array([float(x) for x in o.get("rpy", "0 0 0").split()])
    return xyz, rpy


def rpy_to_matrix(rpy):
    r, pi_, y = rpy
    cx, sx = np.cos(r), np.sin(r)
    cy, sy = np.cos(pi_), np.sin(pi_)
    cz, sz = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def load_link_mesh_verts(link_elem):
    """Return mesh vertices in LINK-LOCAL frame, or None."""
    coll = link_elem.find("collision")
    if coll is None: return None
    geom = coll.find("geometry")
    if geom is None: return None
    mesh = geom.find("mesh")
    if mesh is None: return None
    fname = mesh.get("filename", "")
    scale = np.array([float(x) for x in mesh.get("scale", "1 1 1").split()])
    path = os.path.normpath(os.path.join(URDF_DIR, fname))
    if not os.path.isfile(path): return None
    try:
        m = trimesh.load(path, force='mesh')
    except Exception:
        return None
    verts = np.asarray(m.vertices, dtype=np.float64) * scale
    xyz, rpy = parse_origin(coll)
    R = rpy_to_matrix(rpy)
    return verts @ R.T + xyz


def capsule_distance_to_point(p0, p1, r, q):
    """Distance from point q to the capsule surface (negative = inside)."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    axis = p1 - p0
    L2 = axis @ axis
    if L2 < 1e-12:
        return float(np.linalg.norm(q - p0)) - r
    t = np.clip(((q - p0) @ axis) / L2, 0.0, 1.0)
    closest = p0 + t * axis
    return float(np.linalg.norm(q - closest)) - r


def coverage_check():
    """For each capsule, report mesh coverage statistics."""
    capsules = all_capsule_definitions()
    tree = ET.parse(URDF)
    root = tree.getroot()
    link_meshes = {}
    for link in root.findall("link"):
        verts = load_link_mesh_verts(link)
        if verts is not None:
            link_meshes[link.get("name")] = verts

    print("=" * 80)
    print("PHASE 1: per-link mesh coverage check")
    print("=" * 80)
    print(f"{'link':<32} {'r_now':>7} {'r_req':>7} {'slack':>7} {'cov%':>6}  status")
    print("-" * 80)

    rows = []
    for name, cap in capsules.items():
        if name not in link_meshes:
            continue
        verts = link_meshes[name]
        # signed distance of each vertex to capsule surface
        sd = np.array([capsule_distance_to_point(cap['p0'], cap['p1'], cap['r'], v)
                       for v in verts])
        cov = (sd <= 0).mean()
        max_perp = sd.max() + cap['r']   # absolute max perp dist to axis line
        slack = cap['r'] - max_perp       # positive = capsule has wiggle room
        status = "OK" if cov >= 0.999 else f"UNDER ({(1-cov)*100:.1f}% out)"
        rows.append((name, cap['r'], max_perp, slack, cov, status))

    rows.sort(key=lambda r: r[3])  # sort by slack ascending (under-covered first)
    for name, r_now, r_req, slack, cov, status in rows:
        print(f"{name:<32} {r_now:>7.4f} {r_req:>7.4f} {slack:>+7.4f} "
              f"{cov*100:>5.1f}%  {status}")

    n_under = sum(1 for r in rows if r[4] < 0.999)
    if n_under:
        print(f"\n⚠ {n_under} capsules don't fully cover their mesh - FN risk!")
        print("  Suggested radii (r_req column) would give 100% coverage.")
    else:
        print("\n✓ All capsules fully cover their meshes.")
    return rows


# ----------------------------------------------------------------------
# Phase 2: FP attribution
# ----------------------------------------------------------------------

def setup_pb():
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    flags = p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT
    robot = p.loadURDF(URDF, useFixedBase=True, flags=flags, physicsClientId=cid)

    arm_re = re.compile(r"openarm_(left|right)_joint[1-7]$")
    arm_joints, pos_lims, finger_joints = [], [], []
    name_to_idx = {}
    for i in range(p.getNumJoints(robot, physicsClientId=cid)):
        info = p.getJointInfo(robot, i, physicsClientId=cid)
        jn = info[1].decode()
        ln = info[12].decode()
        name_to_idx[ln] = i
        if info[2] == p.JOINT_REVOLUTE and arm_re.match(jn):
            arm_joints.append(i)
            pos_lims.append((info[8], info[9]))
        if "finger_joint" in jn:
            finger_joints.append(i)
            p.resetJointState(robot, i, 0.01, physicsClientId=cid)

    bl_pairs = build_blacklist_pairs()
    bl_idx = set()
    for fs in bl_pairs:
        a, b = list(fs)
        if a in name_to_idx and b in name_to_idx:
            ia, ib = name_to_idx[a], name_to_idx[b]
            bl_idx.add((min(ia, ib), max(ia, ib)))
            p.setCollisionFilterPair(robot, robot, ia, ib,
                                     enableCollision=0, physicsClientId=cid)
    return cid, robot, arm_joints, finger_joints, pos_lims, name_to_idx, bl_idx, bl_pairs


def fp_attribution(N=1000, range_frac=0.3):
    print()
    print("=" * 80)
    print(f"PHASE 2: false-positive attribution ({N} poses, +-{range_frac*50:.0f}% range)")
    print("=" * 80)

    rng = np.random.default_rng(0)
    cid, robot, arm_joints, finger_joints, pos_lims, name_to_idx, bl_idx, bl_pairs \
        = setup_pb()
    capsules = all_capsule_definitions()
    cap_names = [n for n in capsules if n in name_to_idx]

    lo = np.array([pl[0] for pl in pos_lims])
    hi = np.array([pl[1] for pl in pos_lims])
    span = (hi - lo) * range_frac
    qs = rng.uniform(-span / 2, span / 2, size=(N, len(arm_joints)))
    qs = np.clip(qs, lo, hi)

    # For each pair, track how often it's the minimum-violator on
    # mesh-safe poses (= true FP cause)
    fp_counter = defaultdict(int)
    fp_depth = defaultdict(list)
    safe_pose_count = 0

    for q in qs:
        for jid, a in zip(arm_joints, q):
            p.resetJointState(robot, jid, a, physicsClientId=cid)
        for fj in finger_joints:
            p.resetJointState(robot, fj, 0.01, physicsClientId=cid)
        p.stepSimulation(physicsClientId=cid)

        # Mesh min dist
        pts = p.getClosestPoints(bodyA=robot, bodyB=robot, distance=0.2,
                                  physicsClientId=cid)
        mesh_d = 0.2
        for pt in pts or []:
            a, b = pt[3], pt[4]
            if a == b: continue
            key = (min(a,b), max(a,b))
            if key in bl_idx: continue
            if pt[8] < mesh_d: mesh_d = pt[8]

        if mesh_d <= 0.02:
            continue   # only attribute FP on clearly safe poses
        safe_pose_count += 1

        # Capsule world transforms
        world_caps = {}
        for n in cap_names:
            idx = name_to_idx[n]
            ls = p.getLinkState(robot, idx, computeForwardKinematics=True,
                                 physicsClientId=cid)
            pos = np.array(ls[4])
            R = np.array(p.getMatrixFromQuaternion(ls[5])).reshape(3, 3)
            world_caps[n] = transform_capsule(capsules[n], pos, R)

        # Find the worst (most-violating) capsule pair
        worst_pair, worst_d = None, float("inf")
        for i in range(len(cap_names)):
            for j in range(i+1, len(cap_names)):
                na, nb = cap_names[i], cap_names[j]
                if frozenset((na, nb)) in bl_pairs: continue
                d = capsule_capsule_distance_np(world_caps[na], world_caps[nb])
                if d < worst_d:
                    worst_d = d; worst_pair = (na, nb)
        if worst_pair is not None and worst_d < 0:
            fp_counter[worst_pair] += 1
            fp_depth[worst_pair].append(worst_d)

    print(f"\nOn {safe_pose_count} poses where mesh says safe (>2cm), the most "
          f"-violating capsule pair was:\n")
    print(f"{'count':>6} {'frac':>6} {'med_depth':>10}  pair")
    print("-" * 90)
    rows = []
    for pair, cnt in fp_counter.items():
        rows.append((cnt, cnt/max(1, safe_pose_count), np.median(fp_depth[pair]), pair))
    rows.sort(reverse=True)
    for cnt, frac, med, pair in rows[:20]:
        print(f"{cnt:>6d} {frac*100:>5.1f}% {med:>+10.4f}  {pair[0]} <-> {pair[1]}")

    if not rows:
        print("  (none — capsule never falsely reports collision on safe poses)")
    else:
        print(f"\n→ These pairs are the ones causing capsule FP.")
        print(f"  Each is fixable by:")
        print(f"    a) shrinking r of one of the two capsules")
        print(f"    b) tightening p0/p1 endpoints")
        print(f"    c) blacklisting if they're structurally always overlapping")

    p.disconnect(cid)
    return rows


# ----------------------------------------------------------------------

def main():
    coverage_check()
    fp_attribution(N=1000, range_frac=0.3)


if __name__ == "__main__":
    main()
