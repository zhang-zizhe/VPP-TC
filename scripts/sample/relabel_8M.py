#!/usr/bin/env python3
"""Re-label an existing dataset with the FIXED viability pipeline:
  * qe clamped to joint limits (compute_qe pos_limits),
  * conservative continuous-torque a_max (current DUAL_ACCELERATION_LIMITS).

Only `d_qe` is recomputed (one collision query per row at the new qe).  `dist_q`
is a_max-independent and already correct, so it is reused; `offender_a/b` are a
property of pose q (not qe) and are kept.  Output schema == input schema, so
downstream training/eval needs no change -- it just sees corrected labels.

Parallel (fork + per-worker PyBullet DIRECT, copy-on-write shares the big arrays
so nothing huge is pickled).

Usage:
    python3 scripts/sample/relabel_8M.py \
        --in output/openarm_dual_8M.csv \
        --out output/openarm_dual_8M_relabeled.csv --workers 16
    # quick test on a subset first:
    python3 scripts/sample/relabel_8M.py --nrows 100000 --workers 16 \
        --in output/openarm_dual_8M.csv --out /tmp/relabel_test.csv
"""
import os, sys, time, argparse
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
import pandas as pd
import multiprocessing as mp

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts", "sample"))
import sample_dual_openarm_inter_arm as S
from vpptc.utils import compute_qe
from vpptc.utils_openarm import DUAL_ACCELERATION_LIMITS, DUAL_POS_LIMITS

URDF = os.path.join(_ROOT, "assets/urdf/openarm_description/urdf/robot/openarm_bimanual.urdf")
POS = [f"joint_{i}_pos" for i in range(14)]
VEL = [f"joint_{i}_vel" for i in range(14)]
FIN = [f"joint_{i}_final_pos" for i in range(14)]

# Globals shared to workers via fork copy-on-write (set in parent before Pool).
_Q = _QD = _DQ = None
_ENV = None


def _init(radius):
    global _ENV
    # Smaller search radius than the legacy 10m: positive self-distance over
    # non-blacklisted pairs is small near the boundary, so 0.5m gives IDENTICAL
    # label signs + boundary values (verified: 0 sign flips), only caps a few
    # deep-safe (>0.5m) labels (irrelevant for training).  ~10x faster than 10m.
    S._SEARCH_RADIUS = radius
    _ENV = S._setup(URDF)


def _proc(rng):
    i0, i1 = rng
    out = np.empty((i1 - i0, 16), dtype=np.float64)   # 14 qe + d_qe + min_dist
    for k in range(i0, i1):
        qe = np.asarray(compute_qe(
            _Q[k].tolist(), _QD[k].tolist(),
            acc_limits=DUAL_ACCELERATION_LIMITS,
            pos_limits=DUAL_POS_LIMITS))
        dqe, _, _ = S._min_self_dist(_ENV, qe)
        r = k - i0
        out[r, :14] = qe
        out[r, 14] = dqe
        out[r, 15] = min(_DQ[k], dqe)
    return i0, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="output/openarm_dual_8M.csv")
    ap.add_argument("--out", default="output/openarm_dual_8M_relabeled.csv")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--nrows", type=int, default=None, help="limit rows (test)")
    ap.add_argument("--chunk", type=int, default=50000)
    ap.add_argument("--radius", type=float, default=0.5,
                    help="getClosestPoints search radius (m). 0.5=label-exact & "
                         "fast (~32min/8M); 10=legacy bit-exact deep-safe (~9h).")
    args = ap.parse_args()

    print(f"a_max (conservative) in use: "
          f"{[round(a[1],1) for a in DUAL_ACCELERATION_LIMITS[:7]]}")
    print(f"Loading {args.inp} ...")
    df = pd.read_csv(args.inp, nrows=args.nrows)
    N = len(df)
    print(f"  {N:,} rows")

    global _Q, _QD, _DQ
    _Q = df[POS].values.astype(np.float64)
    _QD = df[VEL].values.astype(np.float64)
    _DQ = df["dist_q"].values.astype(np.float64)

    chunks = [(i, min(i + args.chunk, N)) for i in range(0, N, args.chunk)]
    qe_all = np.empty((N, 14)); dqe_all = np.empty(N); md_all = np.empty(N)

    t0 = time.time(); done = 0
    mp.set_start_method("fork", force=True)
    with mp.Pool(args.workers, initializer=_init, initargs=(args.radius,)) as pool:
        for i0, out in pool.imap_unordered(_proc, chunks):
            n = len(out)
            qe_all[i0:i0+n] = out[:, :14]
            dqe_all[i0:i0+n] = out[:, 14]
            md_all[i0:i0+n] = out[:, 15]
            done += n
            el = time.time() - t0
            print(f"  {done:,}/{N:,}  ({100*done/N:4.1f}%)  "
                  f"{el:5.0f}s  eta {el/done*(N-done):5.0f}s", flush=True)

    old_unsafe = float((df["min_dist"].values < 0).mean())
    df[FIN] = qe_all
    df["dist_qe"] = dqe_all
    df["min_dist"] = md_all
    new_unsafe = float((md_all < 0).mean())
    print(f"\nunsafe%:  OLD {100*old_unsafe:.2f}%  ->  NEW {100*new_unsafe:.2f}%")
    print(f"Writing {args.out} ...")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Done in {time.time()-t0:.0f}s.  -> {args.out}")


if __name__ == "__main__":
    main()
