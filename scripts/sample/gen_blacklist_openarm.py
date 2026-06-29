"""Generate a CORRECT self-collision blacklist for OpenArm.

A link pair is blacklisted IFF its signed distance is not a meaningful
collision signal, i.e. one of:
  (1) PERMANENT OVERLAP  -- max signed distance over all reachable configs < 0
      (the two link meshes always intersect; e.g. adjacent links at a joint).
      Including these makes min_dist always negative -> label useless.
  (2) RIGIDLY FIXED      -- no movable joint between the two links, so the
      distance is constant (range ~ 0); carries zero avoidance information
      (e.g. the two gripper fingers, fixed-joint neighbours).
  (3) PARENT-CHILD       -- explicit from the URDF tree (subset of (1)/(2),
      added for robustness against sampling gaps).

A pair whose distance VARIES and can be both >0 (separated) and <0 (colliding)
is a REAL self-collision mode and is NEVER blacklisted -- the model must learn
it.  When borderline, KEEP (do not blacklist): a wrongly-blacklisted pair
blinds the model (collision); a wrongly-kept structural pair only mildly
distorts the label distribution (safe/conservative).

Inter-arm pairs (left-* vs right-*) are never blacklisted (independent chains:
they always vary and can collide -- they are the whole avoidance target).
"""
import os, sys, argparse
import numpy as np
import multiprocessing as mp

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
URDF = os.path.join(_ROOT, "assets", "urdf", "openarm_description",
                    "urdf", "robot", "openarm_bimanual.urdf")
RADIUS = 0.30   # only pairs that ever come within this matter for collision

def _worker(args):
    n, seed = args
    import pybullet as p, pybullet_data, re
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = p.loadURDF(URDF, useFixedBase=True,
                       flags=p.URDF_USE_SELF_COLLISION
                       | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
    arm_re = re.compile(r"openarm_(left|right)_joint[1-7]$")
    arm, lims = [], []
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        if "finger_joint" in info[1].decode():
            p.resetJointState(robot, i, 0.01)
        if info[2] == p.JOINT_REVOLUTE and arm_re.match(info[1].decode()):
            arm.append(i); lims.append((info[8], info[9]))
    lims = np.array(lims)
    rng = np.random.default_rng(seed)
    stats = {}   # (a,b) -> [min, max, count]
    for _ in range(n):
        q = rng.uniform(lims[:, 0], lims[:, 1])
        for jid, v in zip(arm, q):
            p.resetJointState(robot, jid, v)
        p.performCollisionDetection()
        for pt in p.getClosestPoints(robot, robot, RADIUS):
            a, b = pt[3], pt[4]
            if a == b:
                continue
            key = (min(a, b), max(a, b)); d = pt[8]
            s = stats.get(key)
            if s is None:
                stats[key] = [d, d, 1]
            else:
                s[0] = min(s[0], d); s[1] = max(s[1], d); s[2] += 1
    p.disconnect()
    return stats

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--max-neg", type=float, default=-0.002,
                    help="permanent-overlap: blacklist if max signed dist < this")
    ap.add_argument("--const-range", type=float, default=0.002,
                    help="rigidly-fixed: blacklist if (max-min) < this")
    ap.add_argument("--out", default=os.path.join(_ROOT, "vpptc",
                    "blacklist_openarm.py"))
    args = ap.parse_args()

    import pybullet as p, pybullet_data
    cid = p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = p.loadURDF(URDF, useFixedBase=True,
                       flags=p.URDF_USE_SELF_COLLISION
                       | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
    idx2name = {}
    parent_child = set()
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        idx2name[i] = info[12].decode()
        par = info[16]
        if par >= 0:
            parent_child.add((min(par, i), max(par, i)))
    p.disconnect()

    per = args.n // args.workers
    tasks = [(per, 1000 + k) for k in range(args.workers)]
    print(f"sampling {per*args.workers} configs over {args.workers} workers ...",
          flush=True)
    with mp.Pool(args.workers) as pool:
        results = pool.map(_worker, tasks)
    merged = {}
    for st in results:
        for k, (mn, mx, c) in st.items():
            m = merged.get(k)
            if m is None:
                merged[k] = [mn, mx, c]
            else:
                m[0] = min(m[0], mn); m[1] = max(m[1], mx); m[2] += c

    def is_inter_arm(a, b):
        na, nb = idx2name[a], idx2name[b]
        return (("left" in na and "right" in nb)
                or ("right" in na and "left" in nb))

    blacklist = []   # (name_a, name_b, reason)
    kept_real = []   # variable pairs that CAN collide (min<0<max) -> kept
    for (a, b), (mn, mx, c) in merged.items():
        if is_inter_arm(a, b):
            if mn < 0:
                kept_real.append((idx2name[a], idx2name[b], mn, mx))
            continue
        rng_ = mx - mn
        reason = None
        if (a, b) in parent_child:
            reason = "parent-child"
        elif mx < args.max_neg:
            reason = f"permanent-overlap(max={mx*1000:.0f}mm)"
        elif rng_ < args.const_range:
            reason = f"rigid-fixed(d={mn*1000:.0f}mm)"
        if reason:
            blacklist.append((idx2name[a], idx2name[b], reason))
        elif mn < 0:
            kept_real.append((idx2name[a], idx2name[b], mn, mx))
    # add parent-child pairs that never appeared within RADIUS (always far? rare)
    seen = {(min(*[k for k in [a]]), b) for (a, b) in []}  # noop placeholder
    for (a, b) in parent_child:
        if (a, b) not in merged and not is_inter_arm(a, b):
            blacklist.append((idx2name[a], idx2name[b], "parent-child(unseen)"))

    blacklist.sort(key=lambda x: (x[0], x[1]))
    # write file
    with open(args.out, "w") as f:
        f.write('"""CORRECT self-collision blacklist for OpenArm.\n\n')
        f.write("Generated by gen_blacklist_openarm.py.\n")
        f.write("Blacklist = ONLY pairs that carry no collision signal:\n")
        f.write("  permanent-overlap (max signed dist < 0), rigidly-fixed\n")
        f.write("  (constant distance), or parent-child (URDF).\n")
        f.write("Pairs that can separate AND collide are KEPT (model must\n")
        f.write("learn them).  Inter-arm pairs are never blacklisted.\n")
        f.write(f'Samples: {per*args.workers}.\n"""\n\n')
        f.write("BLACKLIST_NAME_PAIRS = [\n")
        for na, nb, reason in blacklist:
            f.write(f'    ("{na}", "{nb}"),  # {reason}\n')
        f.write("]\n")
    print(f"\nblacklist: {len(blacklist)} pairs -> {args.out}")
    print(f"\n=== KEPT real-collision intra/body pairs (min<0<... CAN collide) "
          f"that the OLD blacklist may have wrongly excluded ===")
    kept_real.sort(key=lambda x: x[2])
    for na, nb, mn, mx in kept_real:
        if not (("left" in na and "right" in nb)
                or ("right" in na and "left" in nb)):
            print(f"  d:[{mn*1000:+.0f},{mx*1000:+.0f}]mm  "
                  f"{na.replace('openarm_','')} <-> {nb.replace('openarm_','')}")

if __name__ == "__main__":
    main()
