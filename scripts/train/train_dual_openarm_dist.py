#!/usr/bin/env python3
"""Train GammaRegressor (single-head, Plan A) on OpenArm dual-arm.

Differences from the old dual-head training script:

  * No CE loss, no logits.  Single regression head; classification at
    eval time is `gamma < margin` on the same predicted distance.
  * Asymmetric Huber loss with collision/boundary upweighting (see
    `vpptc.model.asymmetric_huber_loss`) -- replaces MSE which was
    dominated by far-field samples and starved the boundary region.
  * AMP off by default (the old fp16 + small-target combo caused
    underflow noise that masked the real training signal).  Re-enable
    only after you've confirmed fp32 training is healthy.
  * Linear warmup + cosine decay scheduler (the old constant-lr 8e-4
    AdamW was a common source of transformer instability).
  * Boundary-aware oversampling option: optionally downweight far-field
    samples to flatten the target distribution.
  * Multi-threshold evaluation: report recall/FP at margins
    {0, 5mm, 10mm} so you see the safety-vs-availability tradeoff
    directly instead of a single binary number that hid it.

Inputs are the same CSV produced by
`scripts/sample/sample_dual_openarm_dist.py`.

Usage
-----
    python scripts/train/train_dual_openarm_dist.py \\
        --data output/openarm_dual_collision_distance.csv \\
        --epochs 30
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, random_split


class FastTensorLoader:
    """Bypass DataLoader. For in-RAM tensors, batched indexing is 50-100x
    faster than DataLoader's per-sample __getitem__ + collate path.

    Yields (X_batch, d_batch) on the chosen device, optionally shuffled.
    """
    def __init__(self, X, d, batch_size, shuffle=True, device="cuda",
                 weights=None):
        self.X = X
        self.d = d
        self.bs = batch_size
        self.shuffle = shuffle
        self.device = device
        self.N = X.shape[0]
        self.weights = weights  # 1D tensor for weighted sampling, or None

    def __len__(self):
        return (self.N + self.bs - 1) // self.bs

    def __iter__(self):
        if self.weights is not None:
            # Weighted sampling with replacement (matches WeightedRandomSampler)
            idx = torch.multinomial(self.weights, self.N, replacement=True)
        elif self.shuffle:
            idx = torch.randperm(self.N)
        else:
            idx = torch.arange(self.N)
        for i in range(0, self.N, self.bs):
            j = idx[i:i + self.bs]
            yield (self.X[j].to(self.device, non_blocking=True),
                   self.d[j].to(self.device, non_blocking=True))

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
sys.path.insert(0, _PROJECT_ROOT)

from vpptc.model import GammaRegressor, asymmetric_huber_loss


N_JOINTS = 14
INPUT_DIM = N_JOINTS * 2     # q + qd = 28


# =====================================================================
# Dataset
# =====================================================================

class OpenArmDistDataset(Dataset):
    """Loads (q, qd) -> signed min self-collision distance.

    NO CLIPPING: labels are used exactly as sampled.  Deep penetration
    (down to ~-0.15 m, measured) is kept so the model learns it; max_dist
    (tanh squash) is set wide enough (0.20) that these stay in the
    responsive region of tanh rather than saturating.
    """

    def __init__(self, csv_path: str, max_dist: float = 0.20,
                 clip_neg: float = None):
        # Support comma-separated list of CSVs (concatenated).
        # A path whose name contains "inter" is tagged as inter-arm so the
        # training loop can oversample those rows (--inter-arm-boost).
        paths = [s.strip() for s in csv_path.split(",") if s.strip()]
        if len(paths) == 1:
            df = pd.read_csv(paths[0])
            inter_mask = np.full(len(df), "inter" in paths[0].lower(),
                                 dtype=bool)
        else:
            parts = []
            flags = []
            for pth in paths:
                d_part = pd.read_csv(pth, usecols=lambda c: (
                    c.startswith("joint_") and ("_pos" in c or "_vel" in c)
                    and "final" not in c) or c == "min_dist")
                is_inter = "inter" in pth.lower()
                print(f"    [+] {pth}: {len(d_part):,} rows"
                      f"{'  [inter-arm]' if is_inter else ''}")
                parts.append(d_part)
                flags.append(np.full(len(d_part), is_inter, dtype=bool))
            df = pd.concat(parts, ignore_index=True)
            inter_mask = np.concatenate(flags)
            # Shuffle so batches mix sources (keep mask aligned)
            rng = np.random.RandomState(42)
            order = rng.permutation(len(df))
            df = df.iloc[order].reset_index(drop=True)
            inter_mask = inter_mask[order]
        self.is_inter = torch.tensor(inter_mask, dtype=torch.bool)
        self.X = torch.tensor(
            df.iloc[:, 0:INPUT_DIM].values, dtype=torch.float32
        )
        d = df["min_dist"].values.astype(np.float32)
        # NO clipping -- keep deep penetration and full positive range exactly.
        self.d = torch.tensor(d, dtype=torch.float32)

        n_unsafe = int((self.d < 0).sum().item())
        n_safe = int((self.d >= 0).sum().item())
        n_bndry = int((self.d.abs() < 0.02).sum().item())
        n_satur = int((self.d >= max_dist - 1e-6).sum().item())
        self.unsafe_ratio = n_unsafe / max(1, len(self.d))
        print(f"  Dataset: {len(self.X)} samples")
        print(f"    safe   = {n_safe}  ({100*n_safe/len(self.d):5.2f}%)")
        print(f"    unsafe = {n_unsafe}  ({100*n_unsafe/len(self.d):5.2f}%)")
        print(f"    near boundary (|d|<2cm)   = {n_bndry}  "
              f"({100*n_bndry/len(self.d):5.2f}%)")
        print(f"    saturated at max_dist     = {n_satur}  "
              f"({100*n_satur/len(self.d):5.2f}%)")
        print(f"    distance min/mean/max     = "
              f"{d.min():+.4f} / {d.mean():+.4f} / {d.max():+.4f}")
        n_inter = int(self.is_inter.sum().item())
        print(f"    inter-arm rows            = {n_inter}  "
              f"({100*n_inter/len(self.d):5.2f}%)")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.d[idx]


def make_boundary_sampler(dataset: OpenArmDistDataset, boost: float = 4.0,
                          radius: float = 0.02) -> WeightedRandomSampler:
    """WeightedRandomSampler that upweights boundary + collision samples.

    boost=4.0 means a boundary sample is ~4x more likely to appear
    than a far-field sample in each epoch.  Helps when 50%+ of the
    dataset is saturated at max_dist.
    """
    d = dataset.d.numpy()
    w = np.ones_like(d, dtype=np.float64)
    w[np.abs(d) < radius] *= boost
    w[d < 0] *= boost
    sampler = WeightedRandomSampler(weights=w, num_samples=len(d),
                                     replacement=True)
    return sampler


# =====================================================================
# Evaluation
# =====================================================================

@torch.no_grad()
def evaluate(model, loader, device, margins=(0.0, 0.005, 0.01)):
    """Compute regression and (multi-threshold) classification metrics.

    Critical: report RMSE both overall AND restricted to boundary
    (|d|<2cm) -- the old single RMSE number was dominated by
    far-field samples and didn't reflect decision-relevant accuracy.
    """
    model.eval()
    preds, targets = [], []
    for X, d in loader:
        # X, d already on device from FastTensorLoader
        preds.append(model(X).cpu())
        targets.append(d.cpu())
    pred = torch.cat(preds)
    tgt = torch.cat(targets)

    # Regression
    err = pred - tgt
    rmse_all = float((err ** 2).mean().sqrt())
    bnd = tgt.abs() < 0.02
    rmse_bnd = float((err[bnd] ** 2).mean().sqrt()) if bnd.any() else float("nan")
    coll = tgt < 0
    rmse_coll = float((err[coll] ** 2).mean().sqrt()) if coll.any() else float("nan")

    # Classification at multiple margins
    cls = {}
    for m in margins:
        # ground truth: unsafe iff true_d < m  (apply margin to GT, not pred,
        # so we measure "would the controller using this margin be correct?")
        gt_unsafe = tgt < m
        pred_unsafe = pred < m
        tp = (gt_unsafe & pred_unsafe).sum().item()
        fn = (gt_unsafe & ~pred_unsafe).sum().item()
        tn = (~gt_unsafe & ~pred_unsafe).sum().item()
        fp = (~gt_unsafe & pred_unsafe).sum().item()
        rec = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        acc = (tp + tn) / len(tgt)
        cls[m] = dict(rec_unsafe=rec, fp_rate=fpr, acc=acc)

    return dict(
        rmse_all=rmse_all,
        rmse_boundary=rmse_bnd,
        rmse_collision=rmse_coll,
        cls=cls,
    )


# =====================================================================
# Scheduler: linear warmup + cosine decay
# =====================================================================

def make_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =====================================================================
# Main
# =====================================================================

def get_args():
    p = argparse.ArgumentParser(
        description="OpenArm: train GammaRegressor (single-head, Plan A)")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-4,
                   help="Peak LR (lower than old 8e-4 + warmup added)")
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-dist", type=float, default=0.20,
                   help="Only used for --output-mode tanh (legacy). Output "
                        "range cap for the tanh squash.")
    p.add_argument("--output-mode", type=str, default="linear",
                   choices=["linear", "tanh"],
                   help="linear (default) = unbounded output, NO clip, "
                        "constant gradient. tanh = legacy bounded squash "
                        "(soft-clips output + kills deep-penetration gradient).")
    p.add_argument("--collision-weight", type=float, default=10.0)
    p.add_argument("--boundary-radius", type=float, default=0.02)
    p.add_argument("--boundary-weight", type=float, default=3.0)
    p.add_argument("--huber-delta", type=float, default=0.005)
    p.add_argument("--over-pred-weight", type=float, default=1.0,
                   help="Directional safety penalty: extra factor on "
                        "over-prediction residuals (pred>target = lies "
                        "about safety). 1.0 = symmetric (old behaviour); "
                        "3.0 = optimistic errors hurt 3x more.")
    p.add_argument("--boundary-sampler", action="store_true",
                   help="Use WeightedRandomSampler to oversample boundary "
                        "+ collision samples each epoch")
    p.add_argument("--sampler-boost", type=float, default=4.0)
    p.add_argument("--inter-arm-boost", type=float, default=1.0,
                   help="Oversample inter-arm rows (from a CSV whose path "
                        "contains 'inter') by this factor each epoch. "
                        "1.0 = off. Implies weighted sampling.")
    p.add_argument("--amp", action="store_true",
                   help="Enable fp16 AMP.  OFF by default; turn on only "
                        "after fp32 baseline is verified healthy.")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--output", type=str,
                   default="assets/models/gamma_regressor_openarm.pt")
    return p.parse_args()


def main():
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"AMP: {'on' if args.amp else 'off (fp32)'}")
    print()

    print("Loading data ...")
    ds = OpenArmDistDataset(args.data, max_dist=args.max_dist)
    # Auto-tune collision_weight from data if user left default at 10
    # AND actual imbalance is large.
    auto_cw = max(1.0, min(50.0, 1.0 / max(1e-4, ds.unsafe_ratio)))
    if abs(args.collision_weight - 10.0) < 1e-6:
        print(f"  [auto] collision_weight 10.0 -> {auto_cw:.1f}  "
              f"(from unsafe_ratio={ds.unsafe_ratio:.3%})")
        args.collision_weight = auto_cw
    print()

    # Use a deterministic train/test split via torch generator
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(ds), generator=g)
    train_len = int(0.8 * len(ds))
    train_idx = perm[:train_len]
    test_idx = perm[train_len:]

    X_tr, d_tr = ds.X[train_idx], ds.d[train_idx]
    X_te, d_te = ds.X[test_idx], ds.d[test_idx]

    # Pre-move test set to GPU (1M samples * 28 floats * 4B = 112 MB; trivial)
    X_te_gpu = X_te.to(device)
    d_te_gpu = d_te.to(device)

    is_inter_tr = ds.is_inter[train_idx]
    weights = None
    if args.boundary_sampler or args.inter_arm_boost != 1.0:
        w = torch.ones(len(d_tr), dtype=torch.float64)
        if args.boundary_sampler:
            w[d_tr.abs() < args.boundary_radius] *= args.sampler_boost
            w[d_tr < 0] *= args.sampler_boost
        if args.inter_arm_boost != 1.0:
            w[is_inter_tr] *= args.inter_arm_boost
            print(f"  [sampler] inter-arm rows ({int(is_inter_tr.sum())}) "
                  f"boosted x{args.inter_arm_boost}")
        weights = w

    train_loader = FastTensorLoader(X_tr, d_tr, args.batch_size,
                                     shuffle=True, device=device,
                                     weights=weights)
    test_loader = FastTensorLoader(X_te_gpu, d_te_gpu, args.batch_size,
                                    shuffle=False, device=device)
    print(f"  Train: {train_len}  Test: {len(ds) - train_len}  "
          f"(batched indexing, no DataLoader)")
    print()

    model = GammaRegressor(
        input_dim=INPUT_DIM,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_dist=args.max_dist,
        output_mode=args.output_mode,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: GammaRegressor  "
          f"d_model={args.d_model} nhead={args.nhead} "
          f"layers={args.num_layers}  out={args.output_mode}  "
          f"params={n_params:,}")
    print()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(X_tr) / args.batch_size)
    scheduler = make_scheduler(
        optimizer,
        warmup_steps=args.warmup_epochs * steps_per_epoch,
        total_steps=args.epochs * steps_per_epoch,
    )
    scaler = torch.amp.GradScaler("cuda") if (args.amp and device.type == "cuda") else None

    log = {"loss": [], "rmse_all": [], "rmse_boundary": [],
           "rec_unsafe_0": [], "fp_safe_0": [],
           "rec_unsafe_5mm": [], "fp_safe_5mm": []}

    print(f"Training {args.epochs} epochs ...")
    t_start = time.time()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = 0.0
        n_seen = 0
        t_ep = time.time()
        for X, d in train_loader:
            # X, d already on device from FastTensorLoader
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    pred = model(X)
                    loss = asymmetric_huber_loss(
                        pred, d,
                        delta=args.huber_delta,
                        collision_weight=args.collision_weight,
                        boundary_radius=args.boundary_radius,
                        boundary_weight=args.boundary_weight,
                        over_pred_weight=args.over_pred_weight,
                    )
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(X)
                loss = asymmetric_huber_loss(
                    pred, d,
                    delta=args.huber_delta,
                    collision_weight=args.collision_weight,
                    boundary_radius=args.boundary_radius,
                    boundary_weight=args.boundary_weight,
                    over_pred_weight=args.over_pred_weight,
                )
                loss.backward()
                # Clip grads -- transformer training stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            global_step += 1

            bs = X.size(0)
            ep_loss += loss.item() * bs
            n_seen += bs

        ep_loss /= max(1, n_seen)
        m = evaluate(model, test_loader, device, margins=(0.0, 0.005, 0.01))
        log["loss"].append(ep_loss)
        log["rmse_all"].append(m["rmse_all"])
        log["rmse_boundary"].append(m["rmse_boundary"])
        log["rec_unsafe_0"].append(m["cls"][0.0]["rec_unsafe"] * 100)
        log["fp_safe_0"].append(m["cls"][0.0]["fp_rate"] * 100)
        log["rec_unsafe_5mm"].append(m["cls"][0.005]["rec_unsafe"] * 100)
        log["fp_safe_5mm"].append(m["cls"][0.005]["fp_rate"] * 100)

        dt = time.time() - t_ep
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"  [{epoch:3d}/{args.epochs}] "
            f"loss={ep_loss:.5f}  lr={lr_now:.2e}  "
            f"RMSE all/bnd/coll={m['rmse_all']*1000:5.2f}/"
            f"{m['rmse_boundary']*1000:5.2f}/"
            f"{m['rmse_collision']*1000:5.2f}mm  "
            f"rec@0={m['cls'][0.0]['rec_unsafe']*100:5.2f}%  "
            f"rec@5mm={m['cls'][0.005]['rec_unsafe']*100:5.2f}%  "
            f"FP@5mm={m['cls'][0.005]['fp_rate']*100:5.2f}%  "
            f"({dt:.1f}s)"
        )

    elapsed = time.time() - t_start
    print(f"\nTraining complete in {elapsed:.1f}s")

    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(_PROJECT_ROOT, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": dict(input_dim=INPUT_DIM,
                                d_model=args.d_model,
                                nhead=args.nhead,
                                num_layers=args.num_layers,
                                dropout=args.dropout,
                                max_dist=args.max_dist,
                                output_mode=args.output_mode)},
                out_path)
    print(f"Model saved to {out_path}")

    # ---- Plots ----
    epochs = list(range(1, args.epochs + 1))
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    axes[0].plot(epochs, log["loss"], color="navy")
    axes[0].set_title("Train Loss (weighted Huber)")
    axes[0].set_xlabel("Epoch"); axes[0].grid(True, alpha=0.5)

    axes[1].plot(epochs, [v*1000 for v in log["rmse_all"]], label="all", color="grey")
    axes[1].plot(epochs, [v*1000 for v in log["rmse_boundary"]], label="|d|<2cm", color="crimson")
    axes[1].set_title("Test RMSE (mm)")
    axes[1].set_xlabel("Epoch"); axes[1].legend(); axes[1].grid(True, alpha=0.5)

    axes[2].plot(epochs, log["rec_unsafe_0"], label="margin=0", color="orange")
    axes[2].plot(epochs, log["rec_unsafe_5mm"], label="margin=5mm", color="red")
    axes[2].set_title("Recall on unsafe (%)")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylim(0, 101)
    axes[2].legend(); axes[2].grid(True, alpha=0.5)

    axes[3].plot(epochs, log["fp_safe_0"], label="margin=0", color="steelblue")
    axes[3].plot(epochs, log["fp_safe_5mm"], label="margin=5mm", color="navy")
    axes[3].set_title("False positive rate (%)")
    axes[3].set_xlabel("Epoch"); axes[3].set_ylim(0, 101)
    axes[3].legend(); axes[3].grid(True, alpha=0.5)

    plt.tight_layout()
    fig_path = out_path.replace(".pt", "_curves.png")
    plt.savefig(fig_path, dpi=160, bbox_inches="tight")
    print(f"Curves saved to {fig_path}")


if __name__ == "__main__":
    main()
