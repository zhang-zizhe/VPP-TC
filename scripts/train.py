#!/usr/bin/env python3
"""Train the TransformerGamma self-collision predictor.

Reads a CSV dataset of (q, qd, label) samples and trains a lightweight
Transformer model to predict whether a joint-state pair is collision-free.

Usage
-----
    python scripts/train.py --data data/collision_results.csv --epochs 30
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import recall_score
from torch.utils.data import DataLoader, Dataset, random_split

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, os.pardir)
sys.path.insert(0, os.path.abspath(_PROJECT_ROOT))

from vpptc.model import TransformerGamma


# ======================================================================
# Dataset
# ======================================================================

class CollisionDataset(Dataset):
    """Binary classification dataset: joint state -> collision label.

    The CSV is expected to have 14 input columns (7 positions + 7 velocities)
    starting at column 0, and a binary label at *label_col*.
    """

    def __init__(self, csv_path: str, label_col: int = 21):
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(df.iloc[:, 0:14].values, dtype=torch.float32)
        self.y = torch.tensor(df.iloc[:, label_col].values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ======================================================================
# Evaluation
# ======================================================================

@torch.no_grad()
def evaluate(model, loader, device):
    """Return accuracy and recall on the given data loader."""
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
    recall = recall_score(all_labels, all_preds, average="binary", pos_label=0)
    return accuracy, recall


# ======================================================================
# Training loop
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Train TransformerGamma model")
    parser.add_argument("--data", type=str, required=True,
                        help="Path to the collision dataset CSV")
    parser.add_argument("--label-col", type=int, default=21,
                        help="Column index of the binary label (default: 21)")
    parser.add_argument("--epochs", type=int, default=80,
                        help="Number of training epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=2048,
                        help="Batch size (default: 2048)")
    parser.add_argument("--lr", type=float, default=8e-4,
                        help="Learning rate (default: 8e-4)")
    parser.add_argument("--output", type=str, default="transformer_gamma_ablation_46.pt",
                        help="Output model file (default: transformer_gamma.pt)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    ds = CollisionDataset(args.data, label_col=args.label_col)
    train_len = int(0.8 * len(ds))
    train_ds, test_ds = random_split(ds, [train_len, len(ds) - train_len])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Model
    model = TransformerGamma().to(device)
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    loss_log, acc_log, recall_log = [], [], []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimiser.zero_grad()
            with torch.amp.autocast("cuda"):
                logits, _ = model(X)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            epoch_loss += loss.item() * X.size(0)

        epoch_loss /= train_len
        acc, recall = evaluate(model, test_loader, device)
        print(f"[Epoch {epoch:3d}] loss={epoch_loss:.4f}  "
              f"acc={acc * 100:.2f}%  recall={recall * 100:.2f}%")
        loss_log.append(epoch_loss)
        acc_log.append(acc * 100)
        recall_log.append(recall * 100)

    # Save model
    torch.save(model.state_dict(), args.output)
    print(f"Model saved to {args.output}")

    # --- Plots ---
    epochs = list(range(1, args.epochs + 1))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(epochs, loss_log, marker="o", color="crimson")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(epochs, acc_log, marker="o", color="green")
    axes[1].set_title("Test Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    axes[2].plot(epochs, recall_log, marker="o", color="steelblue")
    axes[2].set_title("Test Recall")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Recall (%)")
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig_path = args.output.replace(".pt", "_curves.png")
    plt.savefig(fig_path, dpi=200, bbox_inches="tight")
    print(f"Training curves saved to {fig_path}")
    plt.show()


if __name__ == "__main__":
    main()
