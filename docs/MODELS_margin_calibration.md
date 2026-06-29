# OpenArm Self-Collision Gamma Models — margin calibration series

> 2026-06-28. This work: diagnose "at deployment, gamma fluctuates wildly at a fixed dist" → clean relabel + add a distance-calibration loss on the classifier margin, turning gamma into a calibrated distance.

---

## 1. Model overview

| Model file | Data | Loss | Status | Deployment threshold |
|---|---|---|---|---|
| `gamma_cls_v012_8M_d128.pt` | v012 (clean) | CE only | Old baseline (2026-06-13, not trained this round) | thr5–8 |
| `gamma_cls_meshfix_d128.pt` | meshfix (**dirty**) | CE only | This round | thr8 |
| **`gamma_margin_d128.pt`** ⭐ | clean (clean) | CE + 0.1·MSE(γ, **200**·dist) | Main model this round | **thr12** |
| `gamma_margin_s500_d128.pt` | clean (clean) | CE + 0.1·MSE(γ, **500**·dist) | This round (testing a stronger barrier) | TBD |

All models share the same architecture: `TransformerGamma`, d_model=128 / nhead=4 / num_layers=4 / dim_ff=512 / dropout=0.1, outputs 2 logits, **γ = logit0 − logit1 (safe−unsafe, large = safe)**.

---

## 2. Dataset lineage (all relabeled from v012's 8M configurations)

| Data csv | Generation script | Which pairs min_dist is computed over | Distance signal |
|---|---|---|---|
| `openarm_dual_8M_v012.csv` | (original sampling) | cross-arm + torso + foldable intra-arm, **structural pairs excluded** | Clean, positive-distance 67mm |
| `openarm_dual_8M_meshfix.csv` | `scripts/sample/relabel_meshfix.py` | V-HACD geometry + 36 blacklist, **but all pairs are computed** | **Dirty**: pinned at 15mm by structural pairs such as link1↔link3 |
| `openarm_dual_8M_clean.csv` | `scripts/sample/relabel_clean.py` | V-HACD geometry + 36 blacklist + **51 structural pairs excluded** | Clean, positive-distance 57mm |

**Structural pairs** = pairs on the same arm that "never approach collision" (min>8mm, e.g. link1↔link3 constantly ~15mm). They are harmless for binary classification (never <0), but they pollute the **distance value** → ruining the target of regression/margin calibration (see [[openarm-label-poison-finding]] for the same kind of problem). `relabel_clean.py` uses `scratch_vhacd/structural_excl.json` (51 pairs) to exclude them.

---

## 3. How each model was trained

Common recipe (`fast_train_*.py`): 100 epochs, bs=4096, AdamW lr=2e-4, cosine+warmup3, AMP, torch.compile, 80/20 split (seed42), data resident on GPU.

### ① `gamma_cls_meshfix_d128.pt` — pure classification (dirty data)
- Script `scripts/train/fast_train_cls.py`
- Data `openarm_dual_8M_meshfix.csv`
- **`loss = CrossEntropy(logits, y)`**, y=(min_dist<0)
- Trains only the **sign** of γ, not its magnitude → γ fluctuates and saturates at a fixed dist (the problem model)

### ② `gamma_margin_d128.pt` — margin calibration ⭐ (the one that solved the problem)
- Script `scripts/train/fast_train_margin.py`, parameters `lam=0.1 scale=200`
- Data `openarm_dual_8M_clean.csv` (**must be clean**, otherwise the calibration target is dirty)
- **`loss = CrossEntropy(logits, y) + 0.1·MSE(γ, 200·clean_dist)`**
  - CE preserves the classification decision; MSE pins γ to `200·distance(meters)` (dist=1mm→target 0.2, dist=40mm→target 8)
  - 200 is the scaling: it maps meters (±0.07) onto a logit range of ±14, equivalent to "1 unit of γ = 5mm"
- Command: `python3 scripts/train/fast_train_margin.py output/openarm_dual_8M_clean.csv assets/models/gamma_margin_d128.pt 100 0.1 200`

### ③ `gamma_margin_s500_d128.pt` — same as ②, SCALE 200→500
- Only difference is `scale=500` → target `500·dist`, larger γ magnitude (max ~28 vs ~11) → stronger barrier gradient
- Purpose: test whether 0% safety can be reached at a lower threshold (②'s γ is on the small side, needs thr12 to be safe)
- Command: `python3 scripts/train/fast_train_margin.py output/openarm_dual_8M_clean.csv assets/models/gamma_margin_s500_d128.pt 100 0.1 500`

---

## 4. Results (test set, clean dist)

| | meshfix (CE-only) | **margin (②)** |
|---|---|---|
| γ std at the same dist (0-2mm) | 4.82 | **1.47** (down ~3.5x) |
| corr(γ, dist) | 0.88 (and huge variance) | **0.979** |
| accuracy | 94.5% | **94.5%** (zero cost) |
| γ behavior | jumps around, saturates | **a stable monotonic function of distance** |

**Core result: γ is now calibrated (signed) distance → "γ fluctuates wildly at a fixed dist" is solved.**

Deployment collision rate (②, 30 seeds, eps0.1): thr4 (engage@20mm) 55% / thr8 (@40mm) 4% / **thr12 (@60mm) 0%**.

---

## ⚠️ 4b. Deployment deep-dive (2026-06-28): calibration solved the fluctuation, but the barrier got weaker and deployment got worse

**Core trade-off (counterintuitive, but backed by data)**: gamma's fluctuation and "the classifier logit is a strong barrier" are bound together — CE does not constrain the logit magnitude, which makes its gradient large near collision (strong braking), at the cost of fluctuation at the same dist; **margin calibration pins down the fluctuation, but at the same time pins down that force → the barrier gets weaker → it must engage earlier to stay safe**.

| Config | Brake% | Tracking error L/R | Result |
|---|---|---|---|
| **s200@thr12** (calibrated, safe config) | **100%** | **150 / 377mm** | Survives, but **brakes the whole time and never draws circles** (end-effector 15-38cm from the circle) = safe but useless |
| meshfix@thr8 (CE-only) | 70% | 35 / 14mm | Drew some circles, 0% over 30 seeds at ctrl-dt0.001 |

**Why**: the limit-cycle's two circles overlap (centers 6cm apart / radius 10cm), so while both arms draw circles they stay very close (<60mm). After s200 calibration the barrier is weak → it needs to engage@60mm to be safe → during circle-drawing it is always within that distance → **brakes the whole time**. meshfix only engages@15mm → it can draw more.

**Other explorations (none broke through)**:
- **SCALE is redundant**: s500 (SCALE500) merely scales γ/threshold/eps together; keeping eps0.1 actually makes it weaker (more collisions). Verified: after scaling eps 2.5x, s500 ≈ s200.
- **tanh target failed**: `10·tanh(dist/15mm)` is only steep at <5mm, while at the ~16mm engage point the barrier's slope ≈ linear → useless, rand-init seed3 collides directly. The steep location was chosen wrong (it should be steep at the engage distance 15-40mm, not at 0).
- **Right-arm tracking R error is large for all models** (14-377mm) → this is a **task problem** (the right-arm xy circle is hard to reach), not a model problem.

**eps is not a knob (measured)**: s200@thr3 (engage@15mm) collides at 75%/67%/67% under eps{0.3,0.6,1.0} (barely moves). **Increasing eps does not help** → s200 can neither be safe at low engage (circle-drawing) nor do anything but 100% empty-braking at high engage, **there is no usable operating point**.

### Final conclusion (closing out this exploration line)

| | gamma fluctuation (the §4 ask) | deployment (circle-drawing + safety) |
|---|---|---|
| **s200 (margin calibration)** | ✅ solved (std↓3.5x, corr0.979) | ❌ barrier too weak, no usable operating point |
| **meshfix / v012 (CE-only)** | ❌ fluctuates | ✅ engage@15mm, 0% collision, can draw circles |

**Core insight: gamma fluctuation and "the barrier being strong enough" are bound together.** CE does not constrain the logit magnitude, which makes its gradient large near collision (strong barrier), at the cost of fluctuation at the same dist; **margin calibration pins gamma to distance and removes the fluctuation, but at the same time pins out the barrier's force**. eps/SCALE/tanh all failed to decouple the two.

**Recommendations for the user**:
- **Deployment still uses CE-only (meshfix or v012)** — the fluctuation is a "surface problem", it is not hurting deployment (even with fluctuation it achieves 0% + draws circles);
- **The value of the margin-calibration model (s200)** is in scenarios that "need gamma = true distance" (interpretability, monitoring, or a different control law), not in this current VPP-TC barrier;
- If you insist on "no fluctuation AND a strong barrier": the only way out is **a single nonlinear target that is steep at the engage distance (15-40mm)** (the tanh idea is right, the D0 location was chosen wrong), left for follow-up.

---

## 5. Deployment commands

```bash
# Main model ② (γ=200·dist, thr12=brake 60mm early)
python3 scripts/simulate/simulate_dual_openarm_cls.py \
  --model-path assets/models/gamma_margin_d128.pt --gamma-threshold 12 \
  --sca-eps 0.1 --ctrl-dt 0.001 --rand-init --seed 3

# Real-time viewing (wall clock 1:1, if slow increase ctrl-dt)
... --ctrl-dt 0.015 --realtime ...
```

The threshold is in **calibrated units**: γ=scale·dist. At SCALE=200, thr=8 ↔ dist=40mm; at SCALE=500, thr=8 ↔ dist=16mm. Changing SCALE requires recalibrating the threshold. The checkpoint stores `margin_scale`.

---

## 6. Circle-drawing task controller gains (at deployment, tune the task, not the model)

Rotation speed `--lc-omega` (default 2.0 rad/s, **this is "how fast it moves"**). Tracking gains: `--lc-kd` (velocity tracking 200), `--lc-kpos` (position spring 120), `--lc-alpha` (limit-cycle convergence rate 20), `--lc-kperp` (off-plane stiffness 20). `--realtime`+`--ctrl-dt` is wall-clock real-time (independent of task speed).
