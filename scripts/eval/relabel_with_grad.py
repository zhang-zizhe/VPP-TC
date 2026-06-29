"""Sobolev data: relabel (q, qd) -> viability margin min_dist AND its exact
gradient d(min_dist)/d[q,qd] (28-dim) via the contact Jacobian.

The gradient is what a CBF barrier actually uses, so supervising it (not just
the value) is the fix for the unfaithful-logit-gradient root cause.

gamma = min(d(q), d(qe)),  qe = q + 0.5*qd*|qd|/a_max (braking pose, clamped).
  if d(q) binds:  dgamma/dq = dd/dq,           dgamma/dqd = 0
  if d(qe) binds: dgamma/dq = dd/dqe,          dgamma/dqd = dd/dqe * |qd|/a_max
(dd/dq is the inter-link contact-Jacobian gradient of the min self-distance.)

Output columns: q(14), qd(14), min_dist(1), grad(28)  = 57 cols.
"""
import os, sys, csv, argparse
import numpy as np
import multiprocessing as mp

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "scripts", "sample"))
URDF = os.path.join(_ROOT, "assets", "urdf", "openarm_description",
                    "urdf", "robot", "openarm_bimanual.urdf")

_E = None
def _init():
    global _E
    import pybullet as p, pybullet_data, re
    from vpptc.utils import compute_qe
    from vpptc.utils_openarm import DUAL_ACCELERATION_LIMITS, DUAL_POS_LIMITS
    from vpptc.blacklist_openarm import BLACKLIST_NAME_PAIRS
    cid = p.connect(p.DIRECT); p.setAdditionalSearchPath(pybullet_data.getDataPath())
    robot = p.loadURDF(URDF, useFixedBase=True,
                       flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
    n2i = {}; i2n = {}; arm = []; allmv = []
    arm_re = re.compile(r"openarm_(left|right)_joint[1-7]$")
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i); nm = info[12].decode(); n2i[nm] = i; i2n[i] = nm
        if "finger_joint" in info[1].decode(): p.resetJointState(robot, i, 0.01)
        if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC): allmv.append(i)
        if info[2] == p.JOINT_REVOLUTE and arm_re.match(info[1].decode()): arm.append(i)
    bl = set()
    for na, nb in BLACKLIST_NAME_PAIRS:
        if na in n2i and nb in n2i: bl.add((min(n2i[na], n2i[nb]), max(n2i[na], n2i[nb])))
    _E = dict(p=p, robot=robot, arm=arm, allmv=allmv,
              arm_cols=[allmv.index(j) for j in arm], n_all=len(allmv), bl=bl,
              cqe=compute_qe, ACC=DUAL_ACCELERATION_LIMITS, POS=DUAL_POS_LIMITS,
              accmax=np.array([a[1] for a in DUAL_ACCELERATION_LIMITS]))

def _min_grad(q14):
    """set arm to q14, return (d_min, dd/dq[14]) over non-blacklist pairs."""
    p = _E["p"]; robot = _E["robot"]
    for jid, v in zip(_E["arm"], q14): p.resetJointState(robot, jid, float(v))
    p.performCollisionDetection()
    best_d, best = 1e9, None
    bl = _E["bl"]
    for pt in p.getClosestPoints(robot, robot, 2.0):
        a, b = pt[3], pt[4]
        if a == b or (min(a, b), max(a, b)) in bl: continue
        if pt[8] < best_d: best_d, best = pt[8], pt
    if best is None: return 1e9, np.zeros(14)
    allpos = [p.getJointState(robot, j)[0] for j in _E["allmv"]]
    z = [0.0]*_E["n_all"]
    lA, lB = best[3], best[4]
    pA, pB = np.array(best[5]), np.array(best[6])
    # separation direction = -contactNormalOnB (robust in penetration too;
    # the witness-point difference pB-pA flips sign when d<0).
    nA = -np.array(best[7])
    nn = np.linalg.norm(nA); nA = nA/nn if nn > 1e-9 else nA
    sA = p.getLinkState(robot, lA, computeForwardKinematics=True)
    sB = p.getLinkState(robot, lB, computeForwardKinematics=True)
    iA = p.invertTransform(sA[0], sA[1]); iB = p.invertTransform(sB[0], sB[1])
    laA = p.multiplyTransforms(iA[0], iA[1], pA.tolist(), [0,0,0,1])[0]
    laB = p.multiplyTransforms(iB[0], iB[1], pB.tolist(), [0,0,0,1])[0]
    JA = np.array(p.calculateJacobian(robot, lA, list(laA), allpos, z, z)[0])
    JB = np.array(p.calculateJacobian(robot, lB, list(laB), allpos, z, z)[0])
    return best[8], (nA @ (JB-JA))[_E["arm_cols"]]

def _work(row):
    q = np.array(row[:14], float); qd = np.array(row[14:28], float)
    d_q, gq_q = _min_grad(q.tolist())
    qe = np.asarray(_E["cqe"](q.tolist(), qd.tolist(), acc_limits=_E["ACC"],
                              pos_limits=_E["POS"]), float)
    d_qe, gq_qe = _min_grad(qe.tolist())
    if d_q <= d_qe:
        h, gq, gqd = d_q, gq_q, np.zeros(14)
    else:
        h, gq, gqd = d_qe, gq_qe, gq_qe * (np.abs(qd)/_E["accmax"])
    return list(q) + list(qd) + [h] + list(gq) + list(gqd)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--max", type=int, default=0)
    a = ap.parse_args()
    def gen():
        with open(a.data) as f:
            r = csv.reader(f); next(r); k = 0
            for line in r:
                try: yield np.array(line[:28], float)
                except Exception: continue
                k += 1
                if a.max and k >= a.max: break
    hdr = ([f"q{i}" for i in range(14)] + [f"qd{i}" for i in range(14)]
           + ["min_dist"] + [f"gq{i}" for i in range(14)] + [f"gqd{i}" for i in range(14)])
    n = 0
    with mp.Pool(a.workers, initializer=_init) as pool, open(a.out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr)
        for res in pool.imap(_work, gen(), chunksize=100):
            w.writerow(res); n += 1
            if n % 100000 == 0: print(f"  {n} ...", flush=True)
    print(f"done {n} rows -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
