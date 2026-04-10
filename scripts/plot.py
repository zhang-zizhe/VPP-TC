#!/usr/bin/env python3
"""Visualise VPP-TC simulation results.

Reads a CSV file produced by ``simulate.py`` and generates comparison plots
for distance-to-target and distance-to-obstacle.

Usage
-----
    python scripts/plot.py --input output/run_1234.csv
    python scripts/plot.py --input output/run_1234.csv --output fig.png
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_SCRIPT_DIR, os.pardir)
sys.path.insert(0, os.path.abspath(_PROJECT_ROOT))

plt.rcParams.update({
    "axes.labelsize": 20,
    "axes.titlesize": 24,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})


def main():
    parser = argparse.ArgumentParser(description="Plot VPP-TC simulation results")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the simulation result CSV")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (default: <input>_plot.png)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output DPI (default: 300)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5))

    # --- Distance to target ---
    ax1.plot(df["time"], df["target_dist"], linewidth=2, label="Distance to target")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Distance (m)")
    ax1.set_title("End-Effector to Target")
    ax1.grid(linestyle=":", alpha=0.6)

    # --- Distance to obstacle ---
    if "real_dist" in df.columns:
        ax2.plot(df["time"], df["real_dist"], linewidth=2, label="Ground truth")
    if "pred_dist" in df.columns:
        ax2.plot(df["time"], df["pred_dist"], linewidth=2, label="SDF prediction")
    if "pred_dist_viability" in df.columns:
        ax2.plot(df["time"], df["pred_dist_viability"], linewidth=2,
                 linestyle="--", label="SDF (viability)")
    ax2.axhline(y=0.10, color="green", linestyle="--", linewidth=1.5,
                label="Reactive threshold")
    ax2.axhline(y=0, color="red", linestyle="--", linewidth=1.5,
                label="Collision boundary")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Distance (m)")
    ax2.set_title("Robot to Obstacle")
    ax2.legend(loc="upper right")
    ax2.grid(linestyle=":", alpha=0.6)

    plt.tight_layout()

    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_plot.png"

    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Figure saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
