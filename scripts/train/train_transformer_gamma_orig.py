#!/usr/bin/env python3
"""Train the ORIGINAL VPP-TC TransformerGamma classifier on the new 8M data.

Faithful port of the upstream training script
    https://github.com/zhang-zizhe/VPP-TC  ->  scripts/train.py
kept as a SEPARATE baseline (new name, new file) -- it does NOT reuse the
current GammaRegressor pipeline.  The original recipe is preserved verbatim:
2-class classifier, plain CrossEntropyLoss, AdamW lr=2e-4, AMP, DataLoader,
report accuracy + recall.

Minimal adaptations for the new dual-arm clean-viability dataset:
  * input is 28 columns (14 q + 14 qd) instead of 14 (Panda).
  * the new CSV has no pre-baked label column 21; the binary label is derived
    from the viability label `min_dist`:  unsafe (class 1) iff min_dist < 0,
    safe (class 0) otherwise.  This matches the model's gamma sign convention
    (gamma = logit_safe - logit_unsafe, larger = safer) used by simulate.
  * recall is reported on the UNSAFE class (pos_label=1) -- the safety-critical
    metric -- rather than upstream's pos_label=0.
  * default batch-size raised to 4096 so 8M rows are tractable.

Usage
-----
    python3 scripts/train/train_transformer_gamma_orig.py \\
        --data output/openarm_dual_8M.csv --epochs 30
"""

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader, Dataset, random_split

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, os.pardir, os.pardir))
sys.path.insert(0, _PROJECT_ROOT)

from vpptc.transformer_gamma_orig import TransformerGamma

INPUT_DIM = 28   # 14 q + 14 qd (dual-arm OpenArm)


# ======================================================================
# Dataset
# ======================================================================

class CollisionDataset(Dataset):
    """Binary classification dataset: joint state -> collision label.

    The first ``INPUT_DIM`` columns are [q, qd]; the label is derived from the
    `min_dist` column (viability label):  class 1 = unsafe (min_dist < 0),
    class 0 = safe.  NO clipping is applied to anything (the label is just a
    sign test).
    """

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(df.iloc[:, 0:INPUT_DIM].values, dtype=torch.float32)
        min_dist = df["min_dist"].values.astype(np.float32)
        y = (min_dist < 0).astype(np.int64)             # 1 = unsafe, 0 = safe
        self.y = torch.tensor(y, dtype=torch.long)

        n_unsafe = int(self.y.sum().item())
        n = len(self.y)
        print(f"  Dataset: {n} samples")
        print(f"    safe   (class 0) = {n - n_unsafe}  ({100*(n-n_unsafe)/n:5.2f}%)")
        print(f"    unsafe (class 1) = {n_unsafe}  ({100*n_unsafe/n:5.2f}%)")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ======================================================================
# Evaluation
# ======================================================================

@torch.no_grad()
def evaluate(model, loader, device):
    """Return accuracy and recall-on-unsafe on the given data loader."""
    model.eval()
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device), y.to(device)
        logits, _ = model(X)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    accuracy = (all_preds == all_labels).mean()
    # pos_label=1 = unsafe class: "of all truly-unsafe samples, how many did we
    # flag?" -- the safety-critical metric.
    recall = recall_score(all_labels, all_preds, average="binary",
                          pos_label=1, zero_division=0)
    return accuracy, recall


# ======================================================================
# Training loop
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train ORIGINAL VPP-TC TransformerGamma classifier (baseline)")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the dataset CSV (e.g. output/openarm_dual_8M.csv)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096,
                        help="Batch size (default 4096; upstream used 512)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--d-model", type=int, default=64,
                        help="Original default 64. Use 128 for a size match "
                             "with GammaRegressor v4.")
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=128,
                        help="FFN width. Original=128. Pass 4*d_model (e.g. 512 "
                             "for d_model=128) so a wider model isn't FFN-bottlenecked.")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--cosine", action="store_true",
                        help="Linear-warmup + cosine LR decay (NOT in the "
                             "faithful original, which used constant LR). "
                             "Recommended for long runs (e.g. 90 epochs) so the "
                             "metric settles instead of bouncing on a plateau.")
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="Warmup epochs (only used with --cosine).")
    parser.add_argument("--output", type=str,
                        default="assets/models/transformer_gamma_orig_8M.pt")
    parser.add_argument("--save-every", type=int, default=0,
                        help="also dump an intermediate checkpoint every N epochs "
                             "(0=off) -> <output>_ep<N>.pt.  Lets us find the "
                             "barrier-optimal epoch (acc plateaus but the logit "
                             "sharpens with more training, hurting the barrier).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data
    print("Loading data ...")
    ds = CollisionDataset(args.data)
    g = torch.Generator().manual_seed(42)
    train_len = int(0.8 * len(ds))
    train_ds, test_ds = random_split(ds, [train_len, len(ds) - train_len],
                                     generator=g)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)
    print(f"  Train: {train_len}  Test: {len(ds) - train_len}")

    # Model
    model = TransformerGamma(
        input_dim=INPUT_DIM,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dropout=args.dropout,
        dim_feedforward=args.dim_feedforward,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: TransformerGamma (original)  "
          f"d_model={args.d_model} nhead={args.nhead} "
          f"layers={args.num_layers}  params={n_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    scheduler = None
    if args.cosine:
        steps_per_epoch = len(train_loader)
        warmup = args.warmup_epochs * steps_per_epoch
        total = args.epochs * steps_per_epoch
        def lr_lambda(step):
            if step < warmup:
                return float(step) / max(1, warmup)
            prog = (step - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)
        print(f"  LR schedule: linear warmup {args.warmup_epochs}ep + cosine decay")
    else:
        print(f"  LR schedule: constant {args.lr} (faithful original)")

    loss_log, acc_log, recall_log = [], [], []

    print(f"Training {args.epochs} epochs ...")
    out_path = args.output
    if not os.path.isabs(out_path):
        out_path = os.path.join(_PROJECT_ROOT, out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def _save(path):
        torch.save({"state_dict": model.state_dict(),
                    "config": dict(input_dim=INPUT_DIM, d_model=args.d_model,
                                   nhead=args.nhead, num_layers=args.num_layers,
                                   dropout=args.dropout,
                                   dim_feedforward=args.dim_feedforward),
                    "model_type": "TransformerGamma"}, path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimiser.zero_grad()
            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits, _ = model(X)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimiser)
                scaler.update()
            else:
                logits, _ = model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimiser.step()
            if scheduler is not None:
                scheduler.step()
            epoch_loss += loss.item() * X.size(0)

        epoch_loss /= train_len
        acc, recall = evaluate(model, test_loader, device)
        lr_now = optimiser.param_groups[0]["lr"]
        print(f"[Epoch {epoch:3d}] loss={epoch_loss:.4f}  lr={lr_now:.2e}  "
              f"acc={acc * 100:.2f}%  recall_unsafe={recall * 100:.2f}%")
        loss_log.append(epoch_loss)
        acc_log.append(acc * 100)
        recall_log.append(recall * 100)
        if args.save_every and epoch % args.save_every == 0 and epoch < args.epochs:
            _save(out_path.replace(".pt", f"_ep{epoch}.pt"))
            print(f"  ckpt -> ep{epoch}")

    # Save final model
    _save(out_path)
    print(f"Model saved to {out_path}")

    # --- Plots ---
    epochs = list(range(1, args.epochs + 1))
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(epochs, loss_log, marker="o", color="crimson")
    axes[0].set_title("Training Loss (CrossEntropy)")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[1].plot(epochs, acc_log, marker="o", color="green")
    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[2].plot(epochs, recall_log, marker="o", color="steelblue")
    axes[2].set_title("Test Recall (unsafe)")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Recall (%)")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig_path = out_path.replace(".pt", "_curves.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Training curves saved to {fig_path}")


if __name__ == "__main__":
    main()
