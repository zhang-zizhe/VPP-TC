#!/usr/bin/env python3
"""Probe: does the safety margin Gamma's GRADIENT saturate in deep penetration?

For VPP-TC the controller's barrier uses d(Gamma)/d[q,qd].  A classifier whose
logit margin is used as Gamma can SATURATE (logits -> +-inf on confident
samples), killing the gradient exactly where you most need a push-out (deep
penetration).  This script measures, per model, |d Gamma/d[q,qd]| binned by the
TRUE signed distance, so saturation shows up as a gradient collapse in the
deep-penetration / deep-safe bins relative to the boundary bin.

Loads a random subset of the dataset, computes per-sample Gamma and input-grad
norm (one batched backward gives all per-sample grads), bins by true min_dist.

Usage:
    python3 scripts/eval/probe_gamma_saturation.py \
        --data output/openarm_dual_8M.csv \
        --models assets/models/transformer_gamma_orig_8M_d128.pt \
                 assets/models/gamma_regressor_openarm_v4.pt \
        --n 200000
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
sys.path.insert(0, _ROOT)
from vpptc.transformer_gamma_orig import TransformerGamma
from vpptc.model import GammaRegressor

INPUT_DIM = 28
BINS = [(-1e9, -0.05, "deep pen  d<-50mm"),
        (-0.05, -0.01, "pen  -50..-10mm"),
        (-0.01,  0.01, "BOUNDARY |d|<10mm"),
        ( 0.01,  0.05, "safe  10..50mm"),
        ( 0.05,  1e9,  "far safe d>50mm")]


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ckpt["config"])
    if ckpt.get("model_type") == "TransformerGamma":
        m = TransformerGamma(**cfg)
        kind = "classifier(logit)"
    else:
        cfg.setdefault("output_mode", "tanh")
        m = GammaRegressor(**cfg)
        kind = f"regressor({cfg.get('output_mode')})"
    m.load_state_dict(ckpt["state_dict"])
    return m.to(device).eval(), kind


def gamma_of(model, x):
    out = model(x)
    return out[1] if isinstance(out, tuple) else out   # classifier->(logits,gamma)


@torch.enable_grad()
def gamma_and_gradnorm(model, X, device, bs=8192):
    gammas, gnorms = [], []
    for i in range(0, len(X), bs):
        xb = X[i:i+bs].to(device).clone().requires_grad_(True)
        g = gamma_of(model, xb)
        gi = torch.autograd.grad(g.sum(), xb)[0]        # per-sample input grads
        gammas.append(g.detach().cpu())
        gnorms.append(gi.norm(dim=1).detach().cpu())    # L2 over 28 dims
    return torch.cat(gammas).numpy(), torch.cat(gnorms).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="output/openarm_dual_8M.csv")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="assets/models/probe_gamma_saturation.png")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {args.data} (cols: 28 feats + min_dist) ...")
    cols = [f"joint_{i}_{p}" for p in ("pos",) for i in range(14)] + \
           [f"joint_{i}_vel" for i in range(14)] + ["min_dist"]
    df = pd.read_csv(args.data, usecols=lambda c: c in set(cols))
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(df), size=min(args.n, len(df)), replace=False)
    sub = df.iloc[idx]
    X = torch.tensor(sub.iloc[:, 0:INPUT_DIM].values, dtype=torch.float32)
    d_true = sub["min_dist"].values.astype(np.float32)
    print(f"  probing {len(X)} samples")
    print(f"  true-dist bin counts:")
    for lo, hi, name in BINS:
        print(f"    {name:22s}: {int(((d_true>=lo)&(d_true<hi)).sum()):>8,}")

    results = {}
    for path in args.models:
        model, kind = load_model(path, device)
        gamma, gnorm = gamma_and_gradnorm(model, X, device)
        results[os.path.basename(path)] = (kind, gamma, gnorm)
        print(f"\n=== {os.path.basename(path)}  [{kind}] ===")
        print(f"  {'bin':22s} {'N':>8s} {'mean Gamma':>11s} "
              f"{'mean|grad|':>11s} {'grad/boundary':>13s}")
        # boundary-bin gradient for normalization
        bm = (d_true >= -0.01) & (d_true < 0.01)
        gbnd = gnorm[bm].mean() if bm.any() else float("nan")
        for lo, hi, name in BINS:
            m = (d_true >= lo) & (d_true < hi)
            if not m.any():
                continue
            print(f"  {name:22s} {int(m.sum()):>8,} {gamma[m].mean():>11.4f} "
                  f"{gnorm[m].mean():>11.5f} {gnorm[m].mean()/gbnd:>13.2f}")

    # scatter: Gamma vs true dist
    nplot = min(20000, len(X))
    pj = rng.choice(len(X), size=nplot, replace=False)
    fig, axes = plt.subplots(1, len(results), figsize=(7*len(results), 5),
                             squeeze=False)
    for ax, (name, (kind, gamma, gnorm)) in zip(axes[0], results.items()):
        ax.scatter(d_true[pj]*1000, gamma[pj], s=2, alpha=0.15)
        ax.axvline(0, color="r", lw=0.8); ax.axhline(0, color="k", lw=0.5)
        ax.set_title(f"{name}\n[{kind}]")
        ax.set_xlabel("true min_dist (mm)"); ax.set_ylabel("predicted Gamma")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    op = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
    plt.savefig(op, dpi=140, bbox_inches="tight")
    print(f"\nScatter saved to {op}")


if __name__ == "__main__":
    main()
