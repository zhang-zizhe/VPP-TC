# VPP-TC: Viability-Preserving Planning with Torque Constraints

Official implementation of **"VPP-TC"**, accepted at **ICRA 2026**.

This repository provides a torque-level robot controller that enforces
*viability-preserving* joint acceleration bounds while simultaneously avoiding
both **self-collisions** (via a learned Transformer safety predictor) and
**external obstacles** (via Robot Distance Fields).

---

## Features

- **Viability-preserving acceleration bounds** -- guarantees the robot can
  always be brought to rest within its joint limits.
- **Self-collision avoidance** -- a lightweight Transformer model predicts a
  scalar safety margin *Gamma* and its gradient, enabling real-time reactive
  control.
- **External collision avoidance** -- integrates
  [Robot Distance Fields (RDF)](https://arxiv.org/abs/2307.00533) for fast,
  differentiable signed-distance queries.
- **QP-based torque controller** -- a convex programme (solved via OSQP)
  minimises end-effector force tracking error subject to safety constraints.
- **PyBullet simulation** -- full simulation environment with the Franka Emika
  Panda 7-DOF manipulator.

---

## Project Structure

```
VPP-TC/
├── README.md
├── LICENSE                     # MIT
├── requirements.txt
├── setup.py                    # pip install -e .
│
├── vpptc/                      # Core Python package
│   ├── __init__.py
│   ├── model.py                # TransformerGamma architecture
│   ├── robot.py                # Panda PyBullet wrapper
│   ├── safety.py               # Acceleration bounds & Gamma predictor
│   ├── sdf.py                  # SDF query wrappers (via RDF)
│   └── utils.py                # Utility functions
│
├── third_party/                # Vendored dependencies
│   └── rdf/                    # Robot Distance Fields 
│
├── assets/                     # Static data
│   ├── urdf/                   # Robot & plane URDF files
│   └── models/                 # Pre-trained weights
│
├── scripts/                    # Entry-point scripts
│   ├── sample.py               # Dataset sampling (self-collision labels)
│   ├── train.py                # Model training
│   ├── simulate.py             # Main simulation
│   └── plot.py                 # Result visualisation
│
└── output/                     # Generated at runtime
```

---
### Steps

```bash
# Clone the repository
git clone https://github.com/<your-username>/VPP-TC.git
cd VPP-TC

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Install in editable mode
pip install -e .
```

This installs the `vpptc` package so that `import vpptc` works from anywhere.

---

## Quick Start

The full pipeline consists of four stages: **sample → train → simulate → plot**.

### 1. Sample the self-collision dataset

Generate labelled `(q, qd, qe, collision)` data by randomly sampling joint
configurations and checking for self-collisions in PyBullet (headless):

```bash
# Uniform sampling (100k samples)
python scripts/sample.py --n-samples 100000

# With near-limit bias (increases collision ratio)
python scripts/sample.py --n-samples 200000 --limit-sampling --limit-joints 3
```

The output CSV is saved to `output/collision_results.csv` by default.

### 2. Train the Transformer model

```bash
python scripts/train.py \
    --data output/collision_results.csv \
    --epochs 30 \
    --batch-size 512 \
    --lr 2e-4 \
    --output assets/models/transformer_gamma.pt
```

### 3. Run the simulation

```bash
python scripts/simulate.py
```

This launches a PyBullet GUI with the Panda robot, a moving obstacle, and the
VPP-TC controller.  Results are saved to `output/`.

Customise parameters:

```bash
python scripts/simulate.py \
    --duration 10 \
    --stepsize 2e-3 \
    --obstacle-pos 0.0 -0.4 0.5 \
    --obstacle-radius 0.05 \
    --target-pos 0.0 -0.6 0.3 \
    --gamma-threshold 2.5 \
    --sdf-react-dist 0.1 \
    --output-dir output
```

### 4. Visualise results

```bash
python scripts/plot.py --input output/run_<timestamp>.csv
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{vpptc2026,
  title     = {VPP-TC: Viability-Preserving Planning with Torque Constraints},
  author    = {<Authors>},
  booktitle = {IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026},
}
```

---

## Acknowledgements

This project uses the [Robot Distance Fields](https://arxiv.org/abs/2307.00533)
library by the Idiap Research Institute, licensed under the MIT License.  The
vendored copy is located in `third_party/rdf/`.

---

## License

This project is licensed under the [MIT License](LICENSE).
