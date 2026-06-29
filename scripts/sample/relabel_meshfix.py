#!/usr/bin/env python3
"""Relabel min_dist=min(d_q,d_qe) on the 8M dataset using the fixed geometry (V-HACD) + new blacklist (34 pairs).
With live progress (imap chunking + memory mapping, avoids re-reading the CSV).
  python3 scripts/sample/relabel_meshfix.py --workers 14
Output: output/openarm_dual_8M_meshfix.csv (q14,qd14,min_dist)
"""
import os, sys, time, argparse
os.environ['OMP_NUM_THREADS'] = '1'
os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.getcwd())
import numpy as np, pandas as pd, multiprocessing as mp

URDF = 'assets/urdf/openarm_description/urdf/robot/openarm_bimanual.urdf'
SRC = 'output/openarm_dual_8M_v012.csv'
OUT = 'output/openarm_dual_8M_meshfix.csv'
XNPY = 'scratch_vhacd/_relabel_X.npy'
PROG = 'scratch_vhacd/_relabel_progress.txt'
TH = 0.06

_E = None
def _init():
    global _E
    import re, pybullet as p, pybullet_data
    from vpptc.utils import compute_qe
    from vpptc.utils_openarm import DUAL_ACCELERATION_LIMITS, DUAL_POS_LIMITS
    from vpptc.blacklist_openarm import BLACKLIST_NAME_PAIRS
    p.connect(p.DIRECT); p.setAdditionalSearchPath(pybullet_data.getDataPath())
    r = p.loadURDF(URDF, useFixedBase=True, flags=p.URDF_USE_SELF_COLLISION)
    i2n = {i: p.getJointInfo(r, i)[12].decode() for i in range(p.getNumJoints(r))}
    arm = [i for i in range(p.getNumJoints(r)) if re.match(r'openarm_(left|right)_joint[1-7]$', p.getJointInfo(r, i)[1].decode())]
    BL = {frozenset(pr) for pr in BLACKLIST_NAME_PAIRS}
    X = np.load(XNPY, mmap_mode='r')
    _E = dict(p=p, r=r, arm=arm, i2n=i2n, BL=BL, cqe=compute_qe,
              acc=DUAL_ACCELERATION_LIMITS, pos=DUAL_POS_LIMITS, X=X)

def _work(rng):
    s, e = rng; E = _E; p = E['p']; r = E['r']; arm = E['arm']; i2n = E['i2n']; BL = E['BL']; X = E['X']
    def md():
        p.performCollisionDetection(); best = TH
        for c in p.getClosestPoints(r, r, TH):
            a, b = c[3], c[4]
            if a == b or frozenset((i2n[a], i2n[b])) in BL: continue
            if c[8] < best: best = c[8]
        return best
    out = np.empty(e - s, 'float32')
    for idx in range(s, e):
        q = X[idx, :14]; qd = X[idx, 14:28]
        for j, v in zip(arm, q): p.resetJointState(r, j, float(v))
        dq = md()
        qe = np.asarray(E['cqe'](list(q), list(qd), acc_limits=E['acc'], pos_limits=E['pos']))
        for j, v in zip(arm, qe): p.resetJointState(r, j, float(v))
        out[idx - s] = min(dq, md())
    return s, out

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=14); a = ap.parse_args()
    print('Loading q,qd ...', flush=True)
    X = pd.read_csv(SRC, usecols=list(range(28))).values.astype('float32')
    N = len(X); np.save(XNPY, X)
    CH = 50000; ranges = [(i, min(i + CH, N)) for i in range(0, N, CH)]
    print(f'Relabeling {N} rows, {a.workers} workers, {len(ranges)} chunks, threshold {TH*1000:.0f}mm', flush=True)
    md = np.empty(N, 'float32'); done = 0; t0 = time.perf_counter()
    with mp.Pool(a.workers, initializer=_init) as pool:
        for s, out in pool.imap_unordered(_work, ranges):
            md[s:s + len(out)] = out; done += len(out)
            el = time.perf_counter() - t0; eta = el / done * (N - done)
            line = f'{done}/{N} ({done/N*100:.1f}%)  elapsed {el/60:.1f}m  ETA {eta/60:.1f}m  current new collision rate {(md[:done]<0).mean()*100:.1f}%'
            print(line, flush=True); open(PROG, 'w').write(line + '\n')
    cols = [f'joint_{i}_pos' for i in range(14)] + [f'joint_{i}_vel' for i in range(14)] + ['min_dist']
    pd.DataFrame(np.column_stack([X, md]), columns=cols).to_csv(OUT, index=False)
    print(f'Done: {N} rows, {(time.perf_counter()-t0)/60:.1f}m, new collision rate {(md<0).mean()*100:.2f}% -> {OUT}', flush=True)
    open(PROG, 'w').write('DONE\n')
