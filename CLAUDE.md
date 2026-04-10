# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VPP-TC (Viability-Preserving Planning with Torque Constraints) is a torque-level robot controller for the Franka Emika Panda manipulator, accepted at ICRA 2026. It enforces viability-preserving joint acceleration bounds while avoiding self-collisions (via a learned Transformer safety predictor) and external obstacles (via Robot Distance Fields). The repo supports both single-arm (7-DOF) and dual-arm (14-DOF) configurations.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -e .
```

The `vpptc` package requires Python >= 3.9. Key dependencies: PyTorch, PyBullet, CVXPY (with OSQP solver), trimesh, mesh_to_sdf.

## Pipeline

The full workflow is **sample → train → simulate → plot**. Single-arm and dual-arm have parallel scripts.

### Single-arm (7-DOF)

```bash
# 1. Sample collision dataset
python scripts/sample.py --n-samples 100000
python scripts/sample.py --n-samples 200000 --limit-sampling --limit-joints 3

# 2. Train TransformerGamma
python scripts/train.py --data output/collision_results.csv --epochs 30 --batch-size 512 --lr 2e-4 --output assets/models/transformer_gamma.pt

# 3. Run simulation (opens PyBullet GUI)
python scripts/simulate.py

# 4. Plot results
python scripts/plot.py --input output/run_<timestamp>.csv
```

### Dual-arm (14-DOF)

```bash
python scripts/sample_dual.py --n-samples 100000
python scripts/train_dual.py --data output/dual_collision_d0.6m.csv --epochs 80 --output transformer_gamma_dual_d0.6.pt
python scripts/simulate_dual.py
```

### Ablation (position-only labels, no viability)

```bash
python scripts/sample_ablation.py --n-samples 3000000
python scripts/train.py --data output/collision_position_only.csv --output transformer_gamma_ablation.pt
python scripts/simulate_ablation.py --model-path assets/models/transformer_gamma_ablation.pt
```

## Architecture

### Core package (`vpptc/`)

- **`model.py`** — `TransformerGamma`: lightweight Transformer encoder that maps a joint-state vector `[q, qd]` to a scalar safety margin Gamma. Input dim is 14 for single-arm, 28 for dual-arm (set via `input_dim` parameter). Each joint value is independently projected to `d_model=64` via a linear layer, added with learned positional encodings, passed through a 4-layer Transformer encoder, mean-pooled, then classified into 2 logits. Gamma = logit[0] - logit[1]; larger Gamma = safer.

- **`safety.py`** — Viability-preserving acceleration bounds (Algorithms 1–3 from the paper). `compute_joint_acceleration_bounds_vec` intersects position limits, velocity limits, viability limits, and trivial acceleration limits per joint. Also provides `gamma_model()` and `compute_gamma_and_grad()` which evaluate the Transformer and backpropagate to get `d(Gamma)/d([q, qd])` for use as a QP constraint. Uses a lazy-loaded singleton for the model.

- **`sdf.py`** — Wraps the vendored RDF library (`third_party/rdf/`) for external obstacle avoidance. `query_sdf_batch()` returns signed distances and joint-space gradients for multiple query points and configurations. Uses `BPSDF` (basis-point SDF) with 24 basis functions and `PandaLayer` for differentiable FK. Lazy-loads model, PandaLayer, and BPSDF engine.

- **`robot.py`** — `Panda` class: PyBullet wrapper for a single 7-DOF Panda. Handles URDF loading, torque/position/velocity control modes, FK/IK, Jacobian, mass matrix, inverse dynamics, and collision queries. Always connects in GUI mode (`p.GUI`).

- **`utils.py`** — `compute_qe()` predicts stopping positions under max deceleration. `feasible_qdd_region()` finds the box corner maximizing Delta-Gamma. `compute_min_center_distance()` gets ground-truth robot-obstacle distance via PyBullet. `JOINT_ACCELERATION_LIMITS` defines per-joint accel bounds for the Panda.

### Control loop (in simulation scripts)

The QP controller minimizes `||J^T_pinv @ u - fc||^2 + alpha * ||u||^2` subject to:
1. Viability-preserving acceleration bounds (box constraints on `M_inv @ u`)
2. Self-collision safety: `g_eff @ M_inv @ u >= eps - c + g_eff @ M_inv @ tau_id` when Gamma < threshold, where `g_eff = 0.5 * grad_q * dt^2 + grad_qd * dt`
3. If QP is infeasible, `eps` is relaxed in steps of 0.1 down to 1e-3; if still infeasible, falls back to greedy corner selection ("soft" mode)

For single-arm: an additional reactive evasion mode activates when SDF distance < `sdf-react-dist`, directly steering via SDF gradients.

For dual-arm: the `simulate_dual.py` script uses two stacked 3×14 Jacobians and a Hopf-oscillator limit-cycle DS for each arm's task-space reference, with shared 14-DOF QP.

### Dual-arm specifics

- URDF: `assets/urdf/panda/panda_dual_arms.urdf` (generated from `assets/Xacro/panda_dual_arms.xacro`), two Pandas at `y=±0.3`
- Collision filter: link5↔link7 pairs within each arm are excluded
- The dual-arm TransformerGamma uses `input_dim=28` (14 positions + 14 velocities)
- Dataset CSV columns: 14 pos + 14 vel + 14 final_pos + 1 label (label at column 42)
- Single-arm CSV columns: 7 pos + 7 vel + 7 final_pos + 1 label (label at column 21)

### Third-party (`third_party/rdf/`)

Vendored copy of Robot Distance Fields (Idiap, MIT license). Key classes: `BPSDF` (basis-point SDF with analytical joint gradients), `PandaLayer` (differentiable FK using STL meshes). Pre-trained models in `third_party/rdf/models/` and `assets/models/rdf/`.

## Key Constants

- Panda joint acceleration limits: `[15, 7.5, 10, 12.5, 15, 20, 20]` (used throughout)
- Panda velocity limits: `[2.175, 2.175, 2.175, 2.175, 2.61, 2.61, 2.61]`
- Default simulation timestep: 2e-3 s; acceleration bound timestep: 0.02 s (hardcoded `dt=0.02` in QP setup)
- Default gamma threshold: 2.5 (single-arm), 4.0 (dual-arm)
- QP solver: OSQP via CVXPY
