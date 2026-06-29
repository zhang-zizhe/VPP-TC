#!/usr/bin/env python3
"""Sweep the decision threshold on Gamma (logit margin) and report how
recall_unsafe / FP-rate / accuracy change.

Decision rule: predict UNSAFE iff Gamma < threshold.  threshold=0 reproduces
the training argmax metric; raising it (toward the sim's avoidance threshold,
e.g. 2.5) flags more states unsafe -> higher recall, higher false alarms.

Usage:
    python3 scripts/eval/sweep_gamma_threshold.py \
        --model assets/models/transformer_gamma_orig_8M_d128_100ep.pt \
        --data output/openarm_dual_8M.csv --n 300000
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _ROOT)
from vpptc.transformer_gamma_orig import TransformerGamma

INPUT_DIM = 28


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="output/openarm_dual_8M.csv")
    ap.add_argument("--n", type=int, default=300000)
    ap.add_argument("--lo", type=float, default=0.0)
    ap.add_argument("--hi", type=float, default=8.0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="assets/models/sweep_gamma_threshold.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.model, map_location=device, weights_only=False)
    model = TransformerGamma(**ckpt["config"]).to(device).eval()
    model.load_state_dict(ckpt["state_dict"])
    print(f"model: {os.path.basename(args.model)}  config={ckpt['config']}")

    cols = [f"joint_{i}_pos" for i in range(14)] + \
           [f"joint_{i}_vel" for i in range(14)] + ["min_dist"]
    df = pd.read_csv(args.data, usecols=lambda c: c in set(cols))
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(df), size=min(args.n, len(df)), replace=False)
    sub = df.iloc[idx]
    X = torch.tensor(sub.iloc[:, 0:INPUT_DIM].values, dtype=torch.float32)
    d_true = sub["min_dist"].values.astype(np.float32)
    true_unsafe = d_true < 0
    n_unsafe = int(true_unsafe.sum()); n_safe = len(d_true) - n_unsafe
    print(f"samples: {len(X):,}  unsafe={n_unsafe:,} ({100*n_unsafe/len(X):.1f}%)  "
          f"safe={n_safe:,}")

    gammas = []
    with torch.no_grad():
        for i in range(0, len(X), 16384):
            _, g = model(X[i:i+16384].to(device))
            gammas.append(g.cpu())
    gamma = torch.cat(gammas).numpy()

    thr = np.arange(args.lo, args.hi + 1e-9, args.step)
    print(f"\n  {'thr(Γ<)':>8s} {'recall_unsafe':>14s} {'FP_rate':>9s} "
          f"{'precision':>10s} {'accuracy':>9s} {'%flagged':>9s}")
    rec_l, fpr_l = [], []
    for t in thr:
        pred = gamma < t
        tp = int((pred & true_unsafe).sum()); fn = n_unsafe - tp
        fp = int((pred & ~true_unsafe).sum()); tn = n_safe - fp
        rec = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        prec = tp / max(1, tp + fp)
        acc = (tp + tn) / len(d_true)
        flagged = pred.mean()
        rec_l.append(rec * 100); fpr_l.append(fpr * 100)
        print(f"  {t:>8.2f} {rec*100:>13.2f}% {fpr*100:>8.2f}% "
              f"{prec*100:>9.2f}% {acc*100:>8.2f}% {flagged*100:>8.1f}%")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thr, rec_l, "o-", color="crimson", label="recall_unsafe (%)")
    ax.plot(thr, fpr_l, "s-", color="steelblue", label="FP rate on safe (%)")
    ax.axvline(2.5, color="grey", ls="--", lw=1, label="sim avoidance thr=2.5")
    ax.set_xlabel("Gamma decision threshold  (predict unsafe iff Γ < thr)")
    ax.set_ylabel("%"); ax.set_ylim(0, 101); ax.grid(True, alpha=0.3); ax.legend()
    ax.set_title(os.path.basename(args.model))
    op = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
    plt.savefig(op, dpi=140, bbox_inches="tight")
    print(f"\nplot saved to {op}")


if __name__ == "__main__":
    main()
