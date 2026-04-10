#!/usr/bin/env python3
"""Train the TransformerGamma collision predictor for a dual-arm OpenArm.

Optimised for RTX 5090 (32GB VRAM):
- Large batch size to saturate GPU
- Cosine annealing LR schedule with warmup
- torch.compile for kernel fusion
- Persistent workers + prefetch for data pipeline
- Mixed precision (fp16)
- Stratified train/val/test split
- Input standardisation
- Class-weighted loss
- Full checkpoint save/resume

Usage
-----
    python scripts/train_dual_openarm.py --data output/openarm_dual_collision_results.csv --epochs 30
    python scripts/train_dual_openarm.py --data ... --resume assets/models/ckpt.pt --epochs 60
"""

import argparse
import math
import os
import random
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import recall_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset, TensorDataset, WeightedRandomSampler

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, os.pardir)
sys.path.insert(0, os.path.abspath(_PROJECT_ROOT))

from vpptc.model import TransformerGamma

N_JOINTS = 14
INPUT_DIM = N_JOINTS * 2


# ======================================================================
# Reproducibility
# ======================================================================

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================================================================
# Evaluation
# ======================================================================

@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool):
    model.eval()
    all_preds, all_labels = [], []
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                logits, _ = model(X)
        else:
            logits, _ = model(X)
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(y.cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    accuracy = (preds == labels).mean()
    recall_col = recall_score(labels, preds, average="binary", pos_label=0)
    recall_safe = recall_score(labels, preds, average="binary", pos_label=1)
    f1 = f1_score(labels, preds, average="macro")
    return accuracy, recall_col, recall_safe, f1


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train TransformerGamma model (OpenArm dual-arm)")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--label-col", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16384,
                        help="Batch size (default: 16384 for 5090)")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Peak learning rate")
    parser.add_argument("--lr-min", type=float, default=1e-5,
                        help="Minimum LR for cosine annealing")
    parser.add_argument("--warmup-epochs", type=int, default=3,
                        help="Linear warmup epochs")
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to full checkpoint to resume from")
    parser.add_argument("--output", type=str,
                        default=os.path.join(
                            _PROJECT_ROOT, "assets", "models",
                            "transformer_gamma_dual_openarm.pt"))
    parser.add_argument("--no-compile", action="store_true",
                        help="Disable torch.compile")
    args = parser.parse_args()

    seed_everything(args.seed)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    use_amp = use_cuda
    print(f"Device: {device}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ------------------------------------------------------------------
    # Data loading + stratified split
    # ------------------------------------------------------------------
    print("Loading data...")
    t0 = time.time()
    df = pd.read_csv(args.data)
    X_all = torch.tensor(df.iloc[:, 0:INPUT_DIM].values, dtype=torch.float32)
    y_all = torch.tensor(df.iloc[:, args.label_col].values, dtype=torch.long)
    print(f"  Loaded {len(X_all)} samples in {time.time()-t0:.1f}s")
    print(f"  Collision rate: {y_all.float().mean():.3f}")

    # Stratified split: 80% train, 10% val, 10% test
    indices = np.arange(len(X_all))
    y_np = y_all.numpy()
    train_idx, temp_idx = train_test_split(
        indices, test_size=0.2, stratify=y_np, random_state=args.seed)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, stratify=y_np[temp_idx], random_state=args.seed)

    # No input standardisation — TransformerGamma's linear_encoder handles
    # per-dimension scaling internally. Removing normalisation preserves
    # correct gradient scaling for the QP avoidance constraint.
    # (Matches the working Panda training scripts.)

    train_ds = TensorDataset(X_all[train_idx], y_all[train_idx])
    val_ds = TensorDataset(X_all[val_idx], y_all[val_idx])
    test_ds = TensorDataset(X_all[test_idx], y_all[test_idx])

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # Class-weighted sampler for training
    train_labels = y_all[train_idx]
    class_counts = torch.bincount(train_labels)
    class_weights = 1.0 / class_counts.float()
    sample_weights = class_weights[train_labels]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=WeightedRandomSampler(sample_weights, len(train_ds), replacement=True),
        num_workers=8, pin_memory=use_cuda, persistent_workers=True,
        prefetch_factor=4, drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=use_cuda, persistent_workers=True)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=4, pin_memory=use_cuda)

    n_batches = len(train_loader)
    samples_per_epoch = n_batches * args.batch_size  # actual samples seen (drop_last)
    print(f"  Batches/epoch: {n_batches}, samples/epoch: {samples_per_epoch}")

    # Class weights for loss
    loss_weights = (class_counts.sum() / (2.0 * class_counts)).float().to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    print(f"  Loss weights: {loss_weights.cpu().tolist()}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = TransformerGamma(input_dim=INPUT_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {n_params:,}")

    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # Cosine annealing with linear warmup
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return args.lr_min / args.lr + (1 - args.lr_min / args.lr) * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    start_epoch = 1
    best_f1 = 0.0
    loss_log, acc_log, recall_col_log, recall_safe_log, f1_log, lr_log = [], [], [], [], [], []

    # ------------------------------------------------------------------
    # Resume from checkpoint
    # ------------------------------------------------------------------
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimiser.load_state_dict(ckpt["optimiser"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", 0.0)
        loss_log = ckpt.get("loss_log", [])
        acc_log = ckpt.get("acc_log", [])
        recall_col_log = ckpt.get("recall_col_log", [])
        recall_safe_log = ckpt.get("recall_safe_log", [])
        f1_log = ckpt.get("f1_log", [])
        lr_log = ckpt.get("lr_log", [])
        print(f"  Resumed from {args.resume}, epoch {start_epoch}, best_f1={best_f1:.4f}")

    # torch.compile
    use_compile = not args.no_compile and hasattr(torch, "compile")
    if use_compile:
        model = torch.compile(model)
        print("  torch.compile enabled")

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n{'Epoch':>6} | {'Loss':>6} | {'Acc':>6} | {'Rec_col':>7}| {'Rec_safe':>8}| {'F1':>6} | {'LR':>8} | {'Time':>5}")
    print("-" * 75)

    for epoch in range(start_epoch, args.epochs + 1):
        t_epoch = time.time()
        model.train()
        epoch_loss = 0.0
        seen_samples = 0

        for X, y in train_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimiser.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    logits, _ = model(X)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimiser)
                scaler.update()
            else:
                logits, _ = model(X)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimiser.step()
            epoch_loss += loss.item() * X.size(0)
            seen_samples += X.size(0)

        epoch_loss /= seen_samples
        scheduler.step()
        current_lr = optimiser.param_groups[0]["lr"]

        # Evaluate on validation set
        acc, rec_col, rec_safe, f1 = evaluate(model, val_loader, device, use_amp)
        dt = time.time() - t_epoch

        print(f"  {epoch:3d}  | {epoch_loss:.4f} | {acc*100:5.2f}% | {rec_col*100:5.2f}% | {rec_safe*100:6.2f}% | {f1:.4f} | {current_lr:.2e} | {dt:.1f}s")

        loss_log.append(epoch_loss)
        acc_log.append(acc * 100)
        recall_col_log.append(rec_col * 100)
        recall_safe_log.append(rec_safe * 100)
        f1_log.append(f1 * 100)
        lr_log.append(current_lr)

        # Save best model (by val F1)
        if f1 > best_f1:
            best_f1 = f1
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save(raw_model.state_dict(), args.output)

        # Save full checkpoint (for resume)
        ckpt_path = args.output.replace(".pt", "_ckpt.pt")
        raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        torch.save({
            "epoch": epoch,
            "model": raw_model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_f1": best_f1,
            "loss_log": loss_log,
            "acc_log": acc_log,
            "recall_col_log": recall_col_log,
            "recall_safe_log": recall_safe_log,
            "f1_log": f1_log,
            "lr_log": lr_log,
        }, ckpt_path)

    # ------------------------------------------------------------------
    # Final evaluation on TEST set
    # ------------------------------------------------------------------
    print(f"\n{'='*75}")
    print("Final evaluation on held-out TEST set:")
    # Load best model
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.load_state_dict(torch.load(args.output, map_location=device, weights_only=True))
    acc, rec_col, rec_safe, f1 = evaluate(model, test_loader, device, use_amp)
    print(f"  Acc={acc*100:.2f}%  Recall_col={rec_col*100:.2f}%  Recall_safe={rec_safe*100:.2f}%  F1={f1*100:.2f}%")

    # ------------------------------------------------------------------
    # Optimal gamma threshold analysis
    # ------------------------------------------------------------------
    print(f"\n--- Gamma threshold analysis (on TEST set) ---")
    print("Convention: gamma > thresh = safe, gamma < thresh = collision")

    # Compute gamma on test set
    test_gammas, test_labels = [], []
    model.eval()
    with torch.no_grad():
        for Xb, yb in test_loader:
            Xb = Xb.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    _, g = model(Xb)
            else:
                _, g = model(Xb)
            test_gammas.append(g.cpu().numpy())
            test_labels.append(yb.numpy())
    test_gammas = np.concatenate(test_gammas)
    test_labels = np.concatenate(test_labels)

    safe_g = test_gammas[test_labels == 0]
    col_g = test_gammas[test_labels == 1]
    print(f"  Safe gamma:      mean={safe_g.mean():+.2f}, std={safe_g.std():.2f}")
    print(f"  Collision gamma: mean={col_g.mean():+.2f}, std={col_g.std():.2f}")

    print(f"\n{'Thresh':>7} | {'Acc':>6} | {'ColRec':>6} | {'SafeRec':>7} | {'F1':>6} | {'Missed':>7}")
    print("-" * 55)
    from sklearn.metrics import recall_score as rs, f1_score as fs
    best_t, best_tf1 = 0, 0
    for t in np.arange(-5, 10, 0.5):
        pred = (test_gammas < t).astype(int)
        a = (pred == test_labels).mean()
        rc = rs(test_labels, pred, pos_label=1)
        rsa = rs(test_labels, pred, pos_label=0)
        f = fs(test_labels, pred, average='macro')
        missed = ((pred == 0) & (test_labels == 1)).sum()
        if f > best_tf1: best_tf1 = f; best_t = t
        if t in [-3,-2,-1,0,1,2,3,4,5] or abs(t - best_t) < 0.6:
            print(f"{t:7.1f} | {a*100:5.1f}% | {rc*100:5.1f}% | {rsa*100:5.1f}% | {f:.4f} | {missed:7d}")
    print(f"\n  Best gamma threshold: {best_t:.1f} (F1={best_tf1:.4f})")
    print(f"  Recommended for sim: thresh={best_t + 1:.1f} to {best_t + 3:.1f} (conservative)")

    print(f"\nBest val F1: {best_f1*100:.2f}%")
    print(f"Model saved to {args.output}")
    print(f"Checkpoint saved to {ckpt_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    epochs = list(range(1, len(loss_log) + 1))
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, loss_log, color="crimson")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, acc_log, color="green", label="Accuracy")
    axes[0, 1].plot(epochs, f1_log, color="purple", label="F1")
    axes[0, 1].set_title("Val Accuracy & F1")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("%")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, recall_col_log, color="steelblue", label="Recall (collision)")
    axes[1, 0].plot(epochs, recall_safe_log, color="orange", label="Recall (safe)")
    axes[1, 0].set_title("Val Recall per Class")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("%")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, lr_log, color="gray")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = args.output.replace(".pt", "_curves.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Training curves saved to {fig_path}")


if __name__ == "__main__":
    main()
