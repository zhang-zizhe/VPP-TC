"""DAgger step-1 diagnostic: re-label deployment danger configs and compare to
the classifier's prediction.  Answers: on the configs the controller actually
drives into (<30mm inter-arm), how wrong is the model -- specifically how many
truly-unsafe configs does it call SAFE (the over-prediction that causes the
collisions)?
"""
import os, sys, csv, argparse
import numpy as np
import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts", "sample"))

from sample_dual_openarm_inter_arm import _setup, _min_self_dist
from vpptc.utils import compute_qe
from vpptc.utils_openarm import DUAL_ACCELERATION_LIMITS, DUAL_POS_LIMITS
from vpptc.transformer_gamma_orig import TransformerGamma

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/tmp/danger_all.csv")
ap.add_argument("--model", default="assets/models/gamma_cls_v012_8M_d128.pt")
ap.add_argument("--thr", type=float, default=5.0)
ap.add_argument("--max", type=int, default=0, help="0=all")
args = ap.parse_args()

URDF = os.path.join(_ROOT, "assets", "urdf", "openarm_description",
                    "urdf", "robot", "openarm_bimanual.urdf")
env = _setup(URDF)

rows = []
with open(args.data) as f:
    r = csv.reader(f); next(r)
    for row in r:
        rows.append(np.array(row[:28], dtype=float))
if args.max:
    rows = rows[::max(1, len(rows)//args.max)][:args.max]
print(f"re-labelling {len(rows)} danger configs ...")

q = np.array([x[:14] for x in rows])
qd = np.array([x[14:28] for x in rows])

true_min, inter_flag = [], []
for i in range(len(rows)):
    d_q, na, nb = _min_self_dist(env, q[i].tolist())
    qe = compute_qe(q[i].tolist(), qd[i].tolist(),
                    acc_limits=DUAL_ACCELERATION_LIMITS,
                    pos_limits=DUAL_POS_LIMITS)
    d_qe, _, _ = _min_self_dist(env, list(qe))
    md = min(d_q, d_qe)
    true_min.append(md)
    inter_flag.append(("left" in na and "right" in nb) or
                      ("right" in na and "left" in nb))
true_min = np.array(true_min); inter_flag = np.array(inter_flag)

# model gamma
ck = torch.load(args.model, map_location="cpu", weights_only=False)
m = TransformerGamma(**ck["config"]).eval(); m.load_state_dict(ck["state_dict"])
X = torch.tensor(np.concatenate([q, qd], axis=1), dtype=torch.float32)
gam = []
with torch.no_grad():
    for i in range(0, len(X), 8192):
        _, g = m(X[i:i+8192]); gam.append(g.numpy())
gam = np.concatenate(gam)

unsafe = true_min < 0
print(f"\n=== TRUE labels on deployment danger configs ===")
print(f"  truly unsafe (min_dist<0) : {unsafe.sum()} ({unsafe.mean()*100:.1f}%)")
print(f"  near-boundary |d|<5mm     : {(np.abs(true_min)<0.005).mean()*100:.1f}%")
print(f"  of unsafe, inter-arm      : {inter_flag[unsafe].mean()*100:.1f}%")
print(f"\n=== MODEL (gamma logit, threshold {args.thr}) ===")
print(f"  gamma min/mean/max        : {gam.min():+.2f} / {gam.mean():+.2f} / {gam.max():+.2f}")
flagged = gam < args.thr
if unsafe.sum():
    miss = unsafe & (gam >= args.thr)          # truly unsafe but model says SAFE
    print(f"  recall on unsafe (gamma<{args.thr}) : "
          f"{(unsafe & flagged).sum()/unsafe.sum()*100:.1f}%")
    print(f"  >>> MISSED (unsafe but gamma>={args.thr}=SAFE): "
          f"{miss.sum()} ({miss.sum()/unsafe.sum()*100:.1f}% of unsafe)")
    over = (gam[unsafe] >= args.thr).mean()*100
    print(f"  these are the over-predictions that let the arms collide")
# also: where true dist is near 0 but model very confident safe
veryconf = (gam > 10) & unsafe
print(f"  unsafe configs the model is VERY confident safe (gamma>10): {veryconf.sum()}")
print(f"\n[compare] training-distribution recall@gamma<5 was 99.98% (sweep).")
