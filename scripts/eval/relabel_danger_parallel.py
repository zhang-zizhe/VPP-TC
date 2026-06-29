"""Parallel re-labeller for collected danger configs (q, qd) -> full 47-col
training rows.  Fork workers, each with its own pybullet env.  Writes the full
relabelled CSV and (optionally deduped) an unsafe-only CSV (min_dist < 0).
"""
import os, sys, csv, argparse
import numpy as np
import multiprocessing as mp

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts", "sample"))

URDF = os.path.join(_ROOT, "assets", "urdf", "openarm_description",
                    "urdf", "robot", "openarm_bimanual.urdf")

_env = None
def _init():
    global _env, _msd, _cqe, _ACC, _POS
    from sample_dual_openarm_inter_arm import _setup, _min_self_dist
    from vpptc.utils import compute_qe
    from vpptc.utils_openarm import DUAL_ACCELERATION_LIMITS, DUAL_POS_LIMITS
    _msd = _min_self_dist; _cqe = compute_qe
    _ACC = DUAL_ACCELERATION_LIMITS; _POS = DUAL_POS_LIMITS
    _env = _setup(URDF)

def _work(row):
    q = row[:14]; qd = row[14:28]
    d_q, na, nb = _msd(_env, list(q))
    qe = np.asarray(_cqe(list(q), list(qd), acc_limits=_ACC, pos_limits=_POS),
                    dtype=float)
    d_qe, na2, nb2 = _msd(_env, list(qe))
    md = min(d_q, d_qe)
    oa, ob = (na2, nb2) if d_qe < d_q else (na, nb)
    return list(q) + list(qd) + list(qe) + [d_q, d_qe, md, oa, ob]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--unsafe-out", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dedup", type=float, default=0.0,
                    help="round q to this (rad) and drop duplicate configs in "
                         "the unsafe set (0 = no dedup)")
    args = ap.parse_args()

    hdr = ([f"joint_{i}_pos" for i in range(14)]
           + [f"joint_{i}_vel" for i in range(14)]
           + [f"joint_{i}_final_pos" for i in range(14)]
           + ["dist_q", "dist_qe", "min_dist", "offender_a", "offender_b"])

    def _row_gen(path):
        # STREAM the input -- never materialise all rows (8M would OOM).
        with open(path) as f:
            r = csv.reader(f); next(r)
            for line in r:
                try:
                    yield np.array(line[:28], dtype=float)
                except Exception:
                    continue

    print(f"relabelling (streaming) with {args.workers} workers ...", flush=True)
    n = 0; n_unsafe = 0
    # only accumulate unsafe rows if --unsafe-out requested (small subset)
    uns = [] if args.unsafe_out else None
    seen = set()
    with mp.Pool(args.workers, initializer=_init) as pool, \
         open(args.out, "w", newline="") as fout:
        w = csv.writer(fout); w.writerow(hdr)
        for res in pool.imap(_work, _row_gen(args.data), chunksize=200):
            w.writerow(res)                      # stream OUT immediately
            n += 1
            if res[44] < 0:
                n_unsafe += 1
                if uns is not None:
                    if args.dedup > 0:
                        key = tuple(np.round(np.array(res[:14]) / args.dedup).astype(int))
                        if key not in seen:
                            seen.add(key); uns.append(res)
                    else:
                        uns.append(res)
            if n % 200000 == 0:
                print(f"  {n} rows ({n_unsafe} unsafe) ...", flush=True)
    print(f"wrote {n} rows ({n_unsafe} unsafe, {n_unsafe/max(n,1)*100:.1f}%) "
          f"-> {args.out}", flush=True)

    if args.unsafe_out:
        with open(args.unsafe_out, "w", newline="") as f:
            ww = csv.writer(f); ww.writerow(hdr); ww.writerows(uns)
        print(f"wrote {len(uns)} unsafe -> {args.unsafe_out}", flush=True)

if __name__ == "__main__":
    main()
