"""Decisive test: at the configs approaching a collision, is avoidance still
PHYSICALLY possible (viable) or already lost (fundamental)?

For each (q, qd): inter-arm distance d, closing rate ddot = (dd/dq).qd, and the
max achievable distance-acceleration dddot_max over the viability accel box.
If dddot_max>0, the controllable braking distance is ddot^2/(2 dddot_max);
the state is VIABLE (avoidable) iff d > that braking distance.
"""
import os, sys, csv
import numpy as np
import pybullet as p, pybullet_data, re

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
from vpptc.utils_openarm import (DUAL_Q_MIN, DUAL_Q_MAX, DUAL_VELOCITY_LIMITS,
                                 DUAL_ACCELERATION_LIMITS)
from vpptc.safety import compute_joint_acceleration_bounds_vec

DATA = sys.argv[1] if len(sys.argv) > 1 else "/tmp/s99_approach.csv"
URDF = os.path.join(_ROOT, "assets", "urdf", "openarm_description",
                    "urdf", "robot", "openarm_bimanual.urdf")
p.connect(p.DIRECT); p.setAdditionalSearchPath(pybullet_data.getDataPath())
robot = p.loadURDF(URDF, useFixedBase=True,
                   flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT)
i2n = {}; arm = []; left_links = []; right_links = []; allmv = []
arm_re = re.compile(r"openarm_(left|right)_joint[1-7]$")
for i in range(p.getNumJoints(robot)):
    info = p.getJointInfo(robot, i); nm = info[12].decode(); i2n[i] = nm
    if "finger_joint" in info[1].decode(): p.resetJointState(robot, i, 0.01)
    if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC): allmv.append(i)
    if info[2] == p.JOINT_REVOLUTE and arm_re.match(info[1].decode()): arm.append(i)
    if nm.startswith("openarm_left_"): left_links.append(i)
    elif nm.startswith("openarm_right_"): right_links.append(i)
arm_cols = [allmv.index(j) for j in arm]
n_all = len(allmv); zeros = [0.0]*n_all
acc_max = np.array([a[1] for a in DUAL_ACCELERATION_LIMITS])
qd_lim = np.array(DUAL_VELOCITY_LIMITS); qmin = np.array(DUAL_Q_MIN); qmax = np.array(DUAL_Q_MAX)

def dist_grad(allpos):
    best_d, best = 1e9, None
    for l1 in left_links:
        for l2 in right_links:
            for pt in p.getClosestPoints(robot, robot, 2.0, l1, l2):
                if pt[8] < best_d: best_d, best = pt[8], pt
    if best is None: return 1e9, np.zeros(14)
    lA, lB = best[3], best[4]
    pA, pB, nB = np.array(best[5]), np.array(best[6]), np.array(best[7])
    sep = pB-pA; ns = np.linalg.norm(sep); nA = sep/ns if ns > 1e-9 else -nB
    sA = p.getLinkState(robot, lA, computeForwardKinematics=True)
    sB = p.getLinkState(robot, lB, computeForwardKinematics=True)
    iA = p.invertTransform(sA[0], sA[1]); iB = p.invertTransform(sB[0], sB[1])
    laA = p.multiplyTransforms(iA[0], iA[1], pA.tolist(), [0,0,0,1])[0]
    laB = p.multiplyTransforms(iB[0], iB[1], pB.tolist(), [0,0,0,1])[0]
    JA = np.array(p.calculateJacobian(robot, lA, list(laA), allpos, zeros, zeros)[0])
    JB = np.array(p.calculateJacobian(robot, lB, list(laB), allpos, zeros, zeros)[0])
    return best[8], (nA @ (JB-JA))[arm_cols]

rows = []
with open(DATA) as f:
    r = csv.reader(f); next(r)
    for line in r: rows.append(np.array(line[:28], float))
rows = rows[-40:]   # last 40 approach steps (closest to collision)
print(f"Analyzing the last {len(rows)} approach configurations (closest to collision):")
print(f"{'d(mm)':>7}{'ddot(m/s)':>11}{'dddot_max':>11}{'brake_dist(mm)':>11}{'avoidable?':>9}")
n_viable = 0
for x in rows:
    q = x[:14]; qd = x[14:28]
    for jid, v in zip(arm, q): p.resetJointState(robot, jid, float(v))
    p.performCollisionDetection()
    allpos = q.tolist()  # arm joints; fingers fixed -> use full not needed for jac cols
    allpos_full = [p.getJointState(robot, j)[0] for j in allmv]
    d, gq = dist_grad(allpos_full)
    ddot = float(gq @ qd)                      # closing rate (neg=approaching)
    lb, ub = compute_joint_acceleration_bounds_vec(q, qd, qmin, qmax, qd_lim,
                                                   acc_max, dt=0.02, viability=True)
    dddot_max = float(np.sum(np.where(gq > 0, gq*ub, gq*lb)))  # max distance accel
    if dddot_max > 1e-6 and ddot < 0:
        brake = ddot**2 / (2*dddot_max)
        viable = d > brake
    elif ddot >= 0:
        brake = 0.0; viable = True
    else:
        brake = float('inf'); viable = False
    n_viable += viable
    print(f"{d*1000:7.1f}{ddot:11.3f}{dddot_max:11.2f}{brake*1000:11.1f}{'  avoidable' if viable else '  unavoidable':>9}")
print(f"\navoidable {n_viable}/{len(rows)}")
print("Conclusion: if the last few steps before collision are mostly 'avoidable' -> the barrier is being used poorly (fixable); mostly 'unavoidable' -> task dead zone")
