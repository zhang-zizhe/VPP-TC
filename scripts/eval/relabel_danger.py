"""DAgger step-2: re-label the collected danger configs (q, qd) into the full
47-column training format (same as the sampler output), so they can be merged
(oversampled) into the 8M and the classifier retrained on them.
"""
import os, sys, csv, argparse
import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts", "sample"))

from sample_dual_openarm_inter_arm import _setup, _min_self_dist
from vpptc.utils import compute_qe
from vpptc.utils_openarm import DUAL_ACCELERATION_LIMITS, DUAL_POS_LIMITS

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp/danger_all.csv")
ap.add_argument("--out", default="output/openarm_dual_danger_relabel.csv")
args = ap.parse_args()

URDF = os.path.join(_ROOT, "assets", "urdf", "openarm_description",
                    "urdf", "robot", "openarm_bimanual.urdf")
env = _setup(URDF)

rows = []
with open(args.data) as f:
    r = csv.reader(f); next(r)
    for row in r:
        rows.append(np.array(row[:28], dtype=float))
print(f"re-labelling {len(rows)} configs -> {args.out}")

hdr = ([f"joint_{i}_pos" for i in range(14)]
       + [f"joint_{i}_vel" for i in range(14)]
       + [f"joint_{i}_final_pos" for i in range(14)]
       + ["dist_q", "dist_qe", "min_dist", "offender_a", "offender_b"])

n_unsafe = 0
with open(args.out, "w", newline="") as f:
    w = csv.writer(f); w.writerow(hdr)
    for i, x in enumerate(rows):
        q = x[:14]; qd = x[14:28]
        d_q, na, nb = _min_self_dist(env, q.tolist())
        qe = np.asarray(compute_qe(q.tolist(), qd.tolist(),
                        acc_limits=DUAL_ACCELERATION_LIMITS,
                        pos_limits=DUAL_POS_LIMITS), dtype=float)
        d_qe, na2, nb2 = _min_self_dist(env, list(qe))
        md = min(d_q, d_qe)
        if md < 0:
            n_unsafe += 1
        # offender = the pair that set the min
        if d_qe < d_q:
            oa, ob = na2, nb2
        else:
            oa, ob = na, nb
        w.writerow(list(q) + list(qd) + list(qe)
                   + [d_q, d_qe, md, oa, ob])
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{len(rows)} ...", flush=True)

print(f"done. {len(rows)} rows, {n_unsafe} unsafe ({n_unsafe/len(rows)*100:.1f}%)")
