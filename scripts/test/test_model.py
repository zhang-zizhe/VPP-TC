#!/usr/bin/env python3
"""Test the trained GammaRegressor: prediction vs ground-truth comparison.

What this does:
  1. Loads the saved model + test CSV
  2. Runs full inference on test set
  3. Reports: regression accuracy, threshold sweep, calibration
  4. Saves a scatter plot (predicted vs true distance)
  5. Saves a threshold sweep plot (recall vs FP rate)

Use this to decide:
  - Is the model good enough to deploy?
  - What margin should the downstream controller use?
  - Where does the model fail (which pose regions)?

Usage
-----
    python scripts/test/test_model.py
    python scripts/test/test_model.py --margin 0.01     # try 10mm margin
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
sys.path.insert(0, _ROOT)

from vpptc.model import GammaRegressor


N_JOINTS = 14
INPUT_DIM = N_JOINTS * 2


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str,
                   default="assets/models/gamma_regressor_openarm.pt")
    p.add_argument("--data", type=str,
                   default="output/openarm_dual_collision_distance.csv")
    p.add_argument("--n-samples", type=int, default=200000,
                   help="Use first N rows of CSV for speed; -1 = all")
    p.add_argument("--margin", type=float, default=0.005,
                   help="Safety margin (m) for the safe/unsafe decision")
    p.add_argument("--out-dir", type=str, default="output/test_results")
    return p.parse_args()


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load model ----
    ckpt_path = os.path.join(_ROOT, args.model) if not os.path.isabs(args.model) else args.model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    cfg.setdefault("output_mode", "tanh")  # old checkpoints were tanh
    model = GammaRegressor(**cfg).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    print(f"Loaded model: {ckpt_path}")
    print(f"  config: {cfg}")
    print(f"  params: {sum(p.numel() for p in model.parameters()):,}")
    print()

    # ---- Load data ----
    data_path = os.path.join(_ROOT, args.data) if not os.path.isabs(args.data) else args.data
    nrows = None if args.n_samples < 0 else args.n_samples
    df = pd.read_csv(data_path, nrows=nrows)
    X = torch.tensor(df.iloc[:, 0:INPUT_DIM].values, dtype=torch.float32)
    d_true = torch.tensor(df["min_dist"].values, dtype=torch.float32)
    # NO clip on the true label -- evaluate against exact distances. (The
    # model output is bounded to [-max_dist, max_dist] by its own tanh.)
    print(f"Loaded {len(X)} samples from {data_path}")
    print()

    # ---- Inference (batched on GPU) ----
    print("Running inference ...")
    preds = []
    bs = 16384
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i:i+bs].to(device)
            preds.append(model(xb).cpu())
    pred = torch.cat(preds).numpy()
    true = d_true.numpy()

    # ---- 1. Regression stats ----
    err = pred - true
    rmse_all = float(np.sqrt(np.mean(err**2)))
    rmse_bnd = float(np.sqrt(np.mean(err[np.abs(true) < 0.01]**2)))
    rmse_coll = float(np.sqrt(np.mean(err[true < 0]**2)))
    bias = float(err.mean())

    print("=" * 70)
    print("REGRESSION")
    print(f"  RMSE all       : {rmse_all*1000:6.2f} mm")
    print(f"  RMSE boundary  : {rmse_bnd*1000:6.2f} mm  (|true_d|<10mm)")
    print(f"  RMSE collision : {rmse_coll*1000:6.2f} mm  (true_d<0)")
    print(f"  bias (mean err): {bias*1000:+6.2f} mm  "
          f"({'over-predicts' if bias>0 else 'under-predicts'} = "
          f"{'less' if bias>0 else 'more'} conservative)")
    print()

    # ---- 2. Threshold sweep ----
    print("=" * 70)
    print(f"CLASSIFICATION (multiple margins)")
    print(f"  {'margin':>8} {'rec_unsafe':>12} {'FP_safe':>10} {'acc':>8}")
    for m in [-0.005, 0.0, 0.005, 0.01, 0.015, 0.02]:
        gt = true < m       # truly unsafe at this margin
        pr = pred < m
        tp = (gt & pr).sum()
        fn = (gt & ~pr).sum()
        tn = (~gt & ~pr).sum()
        fp = (~gt & pr).sum()
        rec = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        acc = (tp + tn) / len(true)
        print(f"  {m*1000:>+5.1f}mm {rec*100:>11.2f}% {fpr*100:>9.2f}% "
              f"{acc*100:>7.2f}%")
    print()

    # ---- 3. ROC/PR at boundary=0 ----
    from sklearn.metrics import roc_auc_score, average_precision_score
    gt0 = (true < 0).astype(np.int32)
    # use -pred so "more unsafe = higher score"
    auc = roc_auc_score(gt0, -pred)
    ap = average_precision_score(gt0, -pred)
    print(f"AUC-ROC (collision detection): {auc:.4f}")
    print(f"AP   (avg precision)        : {ap:.4f}")
    print()

    # ---- 4. Save plots ----
    os.makedirs(os.path.join(_ROOT, args.out_dir), exist_ok=True)

    # Scatter
    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    # subsample for plotting
    idx = np.random.choice(len(pred), min(20000, len(pred)), replace=False)
    ax[0].scatter(true[idx]*1000, pred[idx]*1000, s=1, alpha=0.2)
    lo, hi = -cfg["max_dist"]*1000 - 5, cfg["max_dist"]*1000 + 5
    ax[0].plot([lo, hi], [lo, hi], "r--", linewidth=1, label="y=x (perfect)")
    ax[0].axvline(0, color="k", linewidth=0.5)
    ax[0].axhline(0, color="k", linewidth=0.5)
    ax[0].set_xlabel("True distance (mm)")
    ax[0].set_ylabel("Predicted distance (mm)")
    ax[0].set_title(f"Predicted vs True (RMSE_all={rmse_all*1000:.1f}mm, "
                    f"bias={bias*1000:+.1f}mm)")
    ax[0].set_xlim(lo, hi); ax[0].set_ylim(lo, hi)
    ax[0].legend(); ax[0].grid(True, alpha=0.3)

    # ROC-style sweep
    margins = np.linspace(-0.01, 0.025, 36)
    recs, fps = [], []
    for m in margins:
        gt = true < m
        pr = pred < m
        rec = (gt & pr).sum() / max(1, gt.sum())
        fpr = (~gt & pr).sum() / max(1, (~gt).sum())
        recs.append(rec); fps.append(fpr)
    ax[1].plot(margins*1000, np.array(recs)*100, "g-", label="recall_unsafe", linewidth=2)
    ax[1].plot(margins*1000, np.array(fps)*100, "r-", label="FP_rate", linewidth=2)
    ax[1].axvline(args.margin*1000, color="b", linestyle="--",
                  label=f"chosen margin={args.margin*1000:.1f}mm")
    ax[1].set_xlabel("Safety margin (mm)")
    ax[1].set_ylabel("Rate (%)")
    ax[1].set_title("Margin sweep: pick where recall is high enough")
    ax[1].legend(); ax[1].grid(True, alpha=0.3)
    ax[1].set_ylim(0, 101)

    fig_path = os.path.join(_ROOT, args.out_dir, "scatter_and_sweep.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    print(f"Plots saved to {fig_path}")
    print()

    # ---- 5. Worst-case errors ----
    print("=" * 70)
    print("WORST 10 ERRORS (where prediction is most wrong)")
    abs_err = np.abs(err)
    worst = np.argsort(-abs_err)[:10]
    print(f"  {'idx':>8} {'true(mm)':>10} {'pred(mm)':>10} {'err(mm)':>10}  "
          f"verdict")
    for i in worst:
        verdict = "MISSED COLL" if (true[i] < 0 and pred[i] > 0) else \
                  "FALSE ALARM" if (true[i] > 0.005 and pred[i] < 0) else \
                  "off"
        print(f"  {i:>8d} {true[i]*1000:>+9.2f} {pred[i]*1000:>+9.2f} "
              f"{err[i]*1000:>+9.2f}  {verdict}")
    print()

    # ---- 6. Recommendation ----
    print("=" * 70)
    print("RECOMMENDATION")
    if rec >= 0.95 and fpr <= 0.10:
        print(f"  Model is OK at margin={args.margin*1000:.0f}mm. "
              f"Recall={rec*100:.1f}% FP={fpr*100:.1f}%.")
    else:
        print(f"  At margin={args.margin*1000:.0f}mm: "
              f"recall={rec*100:.1f}% FP={fpr*100:.1f}%.")
        # find margin needed for 99% recall
        for m, r, f in zip(margins, recs, fps):
            if r >= 0.99:
                print(f"  To hit 99% recall, need margin = {m*1000:.1f}mm "
                      f"(at cost of FP={f*100:.1f}%)")
                break
        else:
            print(f"  Cannot hit 99% recall in margin sweep [-10, +25]mm.")
            print(f"  Model needs more training / capacity / data.")


if __name__ == "__main__":
    main()
