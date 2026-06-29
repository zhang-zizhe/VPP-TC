#!/usr/bin/env python3
"""Controlled test: how does HALVING a_max change the viability labels?

Same (q, qd) states, only a_max changes -> qe changes -> d_qe changes ->
label = min(d_q, d_qe) changes.  d_q is a_max-independent (geometry only).

Reuses the sampler's PyBullet setup + exact distance query so the result is
apples-to-apples with the 8M data.  Reports unsafe%, safe->unsafe flips, qe
binding rate, and braking displacement, for OLD vs NEW(conservative) a_max.
"""
import os, sys, importlib.util
import numpy as np
import pandas as pd

# BLAS cap (many tiny collision/IK ops)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts", "sample"))

import sample_dual_openarm_inter_arm as S      # reuse _setup, _min_self_dist
from vpptc.utils import compute_qe

# OLD (current) per-arm a_max, and NEW conservative (continuous-torque, ~half)
OLD = [70., 80., 80., 75., 120., 120., 120.]
NEW = [35., 40., 27., 25., 51., 51., 51.]
def dual(a): return [(-x, x) for x in (a + a)]   # 14-joint (lo,hi) list
OLD_D, NEW_D = dual(OLD), dual(NEW)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 15000
URDF = os.path.join(_ROOT, "assets", "urdf", "openarm_description",
                    "urdf", "robot", "openarm_bimanual.urdf")
CSV = os.path.join(_ROOT, "output", "openarm_dual_8M.csv")

print(f"Loading {N} random states from 8M ...")
pos = [f"joint_{i}_pos" for i in range(14)]
vel = [f"joint_{i}_vel" for i in range(14)]
want = set(pos + vel + ["dist_q", "dist_qe", "min_dist"])
df = pd.read_csv(CSV, usecols=lambda c: c in want)
rng = np.random.default_rng(0)
idx = rng.choice(len(df), size=min(N, len(df)), replace=False)
sub = df.iloc[idx].reset_index(drop=True)
Q = sub[pos].values.astype(float)
QD = sub[vel].values.astype(float)
dq_csv = sub["dist_q"].values.astype(float)
label_old_csv = sub["min_dist"].values.astype(float)

env = S._setup(URDF)

dq_chk = np.empty(N); dqe_old = np.empty(N); dqe_new = np.empty(N)
brk_old = np.empty(N); brk_new = np.empty(N)
for i in range(N):
    q, qd = Q[i], QD[i]
    dq_chk[i], _, _ = S._min_self_dist(env, q)                 # recompute d_q (sanity)
    qe_o = np.asarray(compute_qe(q.tolist(), qd.tolist(), acc_limits=OLD_D))
    qe_n = np.asarray(compute_qe(q.tolist(), qd.tolist(), acc_limits=NEW_D))
    dqe_old[i], _, _ = S._min_self_dist(env, qe_o)
    dqe_new[i], _, _ = S._min_self_dist(env, qe_n)
    brk_old[i] = np.abs(qe_o - q).max()                        # worst-joint braking angle
    brk_new[i] = np.abs(qe_n - q).max()
    if (i + 1) % 2000 == 0:
        print(f"  {i+1}/{N}")

lab_old = np.minimum(dq_chk, dqe_old)
lab_new = np.minimum(dq_chk, dqe_new)

def pct(x): return 100 * x.mean()
print("\n=== sanity: recomputed d_q vs CSV ===")
print(f"  max |Δd_q| = {np.abs(dq_chk - dq_csv).max()*1000:.3f} mm  "
      f"(should be ~0 -> pipeline matches 8M)")
print(f"  recomputed label_old vs CSV min_dist: max |Δ| = "
      f"{np.abs(lab_old - label_old_csv).max()*1000:.3f} mm")

print("\n=== label shift: OLD a_max -> NEW (half) ===")
print(f"  unsafe% (min_dist<0):   OLD {pct(lab_old<0):5.2f}%   NEW {pct(lab_new<0):5.2f}%   "
      f"(+{pct(lab_new<0)-pct(lab_old<0):.2f} pts)")
flip = (lab_old >= 0) & (lab_new < 0)
print(f"  safe -> unsafe flips:   {int(flip.sum())} / {N}  ({pct(flip):.2f}%)")
print(f"  unsafe -> safe flips:   {int(((lab_old<0)&(lab_new>=0)).sum())} (should be ~0)")
bind_o = (dqe_old < dq_chk); bind_n = (dqe_new < dq_chk)
print(f"  qe binding (d_qe<d_q):  OLD {pct(bind_o):5.2f}%   NEW {pct(bind_n):5.2f}%")
print(f"  worst-joint braking angle (deg):  OLD med {np.degrees(np.median(brk_old)):.2f}  "
      f"NEW med {np.degrees(np.median(brk_new)):.2f}  (NEW longer, more conservative)")
print(f"  label percentiles (mm):")
for p_ in [1, 5, 25, 50]:
    print(f"    p{p_:>2}:  OLD {np.percentile(lab_old,p_)*1000:+7.2f}   "
          f"NEW {np.percentile(lab_new,p_)*1000:+7.2f}")
