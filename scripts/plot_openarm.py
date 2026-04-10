#!/usr/bin/env python3
"""Visualise OpenArm VPP-TC simulation results.

Reads a CSV file produced by ``simulate_openarm.py`` or
``simulate_dual_openarm.py`` and generates comparison plots.

Usage
-----
    python scripts/plot_openarm.py --input output/openarm_run_1234.csv
    python scripts/plot_openarm.py --input output/openarm_dual_run_1234.csv
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
    parser = argparse.ArgumentParser(
        description="Plot OpenArm VPP-TC simulation results")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to the simulation result CSV")
    parser.add_argument("--output", type=str, default=None,
                        help="Output image path (default: <input>_plot.png)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output DPI (default: 300)")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    cols = df.columns.tolist()

    # Detect single-arm vs dual-arm
    is_dual = "inter_arm_dist" in cols

    if is_dual:
        fig, axes = plt.subplots(1, 3, figsize=(24, 5))

        # Gamma
        axes[0].plot(df["time"], df["gamma"], linewidth=2, color="steelblue")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Gamma")
        axes[0].set_title("Self-Collision Safety Margin")
        axes[0].grid(linestyle=":", alpha=0.6)

        # Inter-arm distance
        axes[1].plot(df["time"], df["inter_arm_dist"], linewidth=2,
                     label="Inter-arm distance")
        if "collision_dist" in cols:
            axes[1].plot(df["time"], df["collision_dist"], linewidth=2,
                         linestyle="--", label="Collision distance")
        axes[1].axhline(y=0, color="red", linestyle="--", linewidth=1.5,
                        label="Collision boundary")
        axes[1].set_xlabel("Time (s)")
        axes[1].set_ylabel("Distance (m)")
        axes[1].set_title("Inter-Arm Distance")
        axes[1].legend(loc="upper right")
        axes[1].grid(linestyle=":", alpha=0.6)

        # Limit cycle tracking
        axes[2].plot(df["time"], df["lc_dist_left"], linewidth=2, label="Left arm")
        axes[2].plot(df["time"], df["lc_dist_right"], linewidth=2, label="Right arm")
        axes[2].set_xlabel("Time (s)")
        axes[2].set_ylabel("Radial Error (m)")
        axes[2].set_title("Limit Cycle Tracking")
        axes[2].legend(loc="upper right")
        axes[2].grid(linestyle=":", alpha=0.6)

    else:
        n_plots = sum(1 for c in ["target_dist", "self_collision_dist", "gamma"]
                      if c in cols)
        fig, axes = plt.subplots(1, max(n_plots, 2), figsize=(10 * max(n_plots, 2), 5))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        idx = 0

        if "target_dist" in cols:
            axes[idx].plot(df["time"], df["target_dist"], linewidth=2,
                           label="Distance to target")
            axes[idx].set_xlabel("Time (s)")
            axes[idx].set_ylabel("Distance (m)")
            axes[idx].set_title("End-Effector to Target")
            axes[idx].grid(linestyle=":", alpha=0.6)
            idx += 1

        if "self_collision_dist" in cols:
            axes[idx].plot(df["time"], df["self_collision_dist"], linewidth=2,
                           label="Self-collision distance")
            axes[idx].axhline(y=0, color="red", linestyle="--", linewidth=1.5,
                              label="Collision boundary")
            axes[idx].set_xlabel("Time (s)")
            axes[idx].set_ylabel("Distance (m)")
            axes[idx].set_title("Self-Collision Distance")
            axes[idx].legend(loc="upper right")
            axes[idx].grid(linestyle=":", alpha=0.6)
            idx += 1

        if "gamma" in cols:
            axes[idx].plot(df["time"], df["gamma"], linewidth=2, color="steelblue",
                           label="Gamma")
            axes[idx].set_xlabel("Time (s)")
            axes[idx].set_ylabel("Gamma")
            axes[idx].set_title("Self-Collision Safety Margin")
            axes[idx].grid(linestyle=":", alpha=0.6)
            idx += 1

    plt.tight_layout()

    if args.output is None:
        base = os.path.splitext(args.input)[0]
        args.output = f"{base}_plot.png"

    plt.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    print(f"Figure saved to {args.output}")
    plt.show()


if __name__ == "__main__":
    main()
