#!/usr/bin/env python3
"""Re-label a small batch under 3 configs to see the effect of (a) the qe
limit-clamp and (b) the conservative a_max, separately and combined.

Same (q, qd) states + SAME collision query for all configs -> deltas are clean.
  OLD   : old a_max [70..120], NO clamp     (reproduces the current 8M pipeline)
  CLAMP : old a_max,           clamp         (isolates the clamp effect)
  NEW   : new a_max [35..51],  clamp         (final fixed pipeline)
"""
import os, sys
import numpy as np, pandas as pd, math
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "scripts", "sample"))
import sample_dual_openarm_inter_arm as S
from vpptc.utils import compute_qe
from vpptc.utils_openarm import DUAL_POS_LIMITS

def dual(a): return [(-x, x) for x in (a + a)]
OLD_D = dual([70., 80., 80., 75., 120., 120., 120.])
NEW_D = dual([35., 40., 26.7, 25., 51.4, 51.4, 51.4])
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
URDF = os.path.join(_ROOT, "assets/urdf/openarm_description/urdf/robot/openarm_bimanual.urdf")

pos=[f'joint_{i}_pos' for i in range(14)]; vel=[f'joint_{i}_vel' for i in range(14)]
df = pd.read_csv(os.path.join(_ROOT,"output/openarm_dual_8M.csv"),
                 usecols=lambda c: c in set(pos+vel+["min_dist"]))
rng = np.random.default_rng(0); idx = rng.choice(len(df), N, replace=False)
sub = df.iloc[idx].reset_index(drop=True)
Q=sub[pos].values; QD=sub[vel].values; md_csv=sub["min_dist"].values

env = S._setup(URDF)
dq=np.empty(N); lab={"OLD":np.empty(N),"CLAMP":np.empty(N),"NEW":np.empty(N)}
for i in range(N):
    q,qd = Q[i].tolist(), QD[i].tolist()
    dq[i],_,_ = S._min_self_dist(env, Q[i])
    qe_old   = np.asarray(compute_qe(q,qd,acc_limits=OLD_D))
    qe_clamp = np.asarray(compute_qe(q,qd,acc_limits=OLD_D,pos_limits=DUAL_POS_LIMITS))
    qe_new   = np.asarray(compute_qe(q,qd,acc_limits=NEW_D,pos_limits=DUAL_POS_LIMITS))
    for nm,qe in (("OLD",qe_old),("CLAMP",qe_clamp),("NEW",qe_new)):
        dqe,_,_ = S._min_self_dist(env, qe)
        lab[nm][i] = min(dq[i], dqe)
    if (i+1)%2000==0: print(f"  {i+1}/{N}")

def P(x): return 100*np.mean(x)
print(f"\n=== sanity: my OLD-label vs CSV min_dist ===")
print(f"   mean|Δ|={np.abs(lab['OLD']-md_csv).mean()*1000:.2f}mm  (tiny query difference, delta comparison unaffected)")
print(f"\n=== unsafe% (label<0) ===   N={N}")
for nm in ("OLD","CLAMP","NEW"):
    print(f"   {nm:6}: {P(lab[nm]<0):6.2f}%")
print(f"\n=== flips relative to OLD ===")
for nm in ("CLAMP","NEW"):
    s2u=((lab['OLD']>=0)&(lab[nm]<0)); u2s=((lab['OLD']<0)&(lab[nm]>=0))
    print(f"   {nm:6}: safe->unsafe {int(s2u.sum()):5d} ({P(s2u):.2f}%)   "
          f"unsafe->safe {int(u2s.sum()):4d} ({P(u2s):.2f}%)")
print(f"\n=== label percentiles (mm) ===")
print(f"   {'p':>4} {'OLD':>9} {'CLAMP':>9} {'NEW':>9}")
for pp in [1,5,25,50,75]:
    print(f"   {pp:>3}% {np.percentile(lab['OLD'],pp)*1000:>8.2f} "
          f"{np.percentile(lab['CLAMP'],pp)*1000:>8.2f} {np.percentile(lab['NEW'],pp)*1000:>8.2f}")
