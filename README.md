# OpenArm Dual-Arm Self-Collision Avoidance — VPP-TC (`open-arm` branch)

Self-collision avoidance for the **OpenArm** bimanual robot (14-DOF, 7 per arm) under the
**VPP-TC** framework (Viability-Preserving Planning with Torque Constraints, ICRA 2026).
A Transformer predicts a scalar safety margin **γ**; γ is used as a CBF barrier inside a
torque-level QP controller so the two arms can run a limit-cycle (circle-tracing) task while
avoiding (a) arm-to-arm collisions, (b) arm-to-torso collisions, and (c) single-arm self-folding.

> This branch is the complete OpenArm record on top of VPP-TC: **every problem we hit and how we
> fixed it**, the **usable models**, and the **sampling / training** pipeline. The dataset CSVs
> (~92 GB total) are NOT committed; see §4 for how to regenerate them.

---

## 1. Repository layout

```
vpptc/                       # core package
  transformer_gamma_orig.py    # γ model (2-class classifier; γ = logit0 - logit1 = safety margin)
  blacklist_openarm.py         # self-collision pair blacklist (36 pairs, see §5.2)
  utils_openarm.py             # joint/accel limits, compute_qe (braking pose)
  safety.py / safety_openarm.py# viability accel box + QP barrier
scripts/
  sample/   relabel_clean.py, relabel_meshfix.py, sample_dual_openarm_inter_arm.py,
            gen_blacklist_openarm.py
  train/    fast_train_cls.py (CE classifier), fast_train_margin.py (margin distance calibration)
  simulate/ simulate_dual_openarm_cls.py (dual-arm limit-cycle GUI/headless sim)
  eval/ test/ analysis/ visualize_convex_hulls.py
assets/
  urdf/     OpenArm bimanual URDF + meshes (incl. V-HACD convex-decomposed collision meshes, §6)
  models/   6 usable models (see §7)
docs/MODELS_margin_calibration.md   # detailed record of the margin-calibration experiments
```

Datasets (`output/*.csv`, ~92 GB) and the 117 MB high-res **visual** meshes are excluded by
`.gitignore`. The **collision** meshes (including our V-HACD decompositions) and the 6 models **are**
committed, so headless collision-avoidance (`--no-gui`) and all evaluation run out of the box. For GUI
rendering, download the OpenArm description visual meshes into
`assets/urdf/openarm_description/meshes/*/visual/`.

## 2. Install & quick run

```bash
pip install -e . && pip install -r requirements.txt   # needs pybullet, torch, cvxpy/osqp, pandas
# Simulation (GUI): both arms trace circles + avoid self-collision
python3 scripts/simulate/simulate_dual_openarm_cls.py \
  --model-path assets/models/gamma_cls_v012_8M_d128.pt \
  --gamma-threshold 8 --sca-eps 0.1 --ctrl-dt 0.001 --rand-init --seed 3
# headless: add --no-gui ; real-time viewing: add --realtime (if slow, raise --ctrl-dt e.g. 0.015)
```

## 3. Method (in one paragraph)

The model takes the current state `(q, qd)` (28-dim) and outputs **γ = logit0 − logit1**
(safety margin; larger = safer). The controller triggers avoidance when `γ < threshold` and uses
`d(γ)/d(q,qd)` as the linear barrier direction in the QP:
`g_eff·M⁻¹·u ≥ eps − c + g_eff·M⁻¹·τ_id` with `g_eff = 0.5·∇qγ·dt² + ∇qdγ·dt`, on top of the
viability acceleration box. **Label = `min(d_q, d_qe)`**: the smaller of the min self-collision
distance at the current pose and at the braking-stop pose `qe = compute_qe(q, qd)` (viability);
`< 0` means collision.

## 4. Data sampling

8M dataset = **5M random + 3M targeted** (`sample_dual_openarm_inter_arm.py`):
- **random** — uniform within joint limits;
- **targeted** — IK-biased **inter-arm boundary** sampling (`--require-inter-arm`), to add cross-arm
  near-collision samples (random configs rarely bring the arms close, so inter-arm collisions are scarce).

`min_dist` is taken only over **collidable pairs**: inter-arm (left↔right) + torso (body0↔arm) +
foldable intra-arm pairs (e.g. hand↔link1); it **excludes "never-approaching" intra-arm structural
pairs** (e.g. link1↔link3, constant ~15mm). CSV columns: `q14, qd14, qe14 (final_pos), dist_q,
dist_qe, min_dist, offender_a/b`. Full resample ≈ 16h; after a geometry change, relabel in parallel:

```bash
python3 scripts/sample/relabel_clean.py --workers 14   # → output/openarm_dual_8M_clean.csv
```

## 5. Problems we hit & how we solved them (the important part)

### 5.1 Mesh convex-hull over-detection → V-HACD convex decomposition
**Problem**: PyBullet takes the **convex hull** of every articulated link's collision mesh. OpenArm's
collision meshes are **concave** (`is_convex=False`; hull/true volume ratio: body0 **5.56×**,
link5 4.65×, hand 3.73×, link4 2.64×…), so concavities get filled → **systematic over-detection**
(hull surface bulges ~57mm median beyond the true mesh). Consequence: **~25% of inter-arm collision
labels are phantom**; body0 appears in ~15% of collision pairs.
**Fix**: **V-HACD convex decomposition** of the concave meshes (multi-piece COMPOUND respects
concavity); point the URDF `<collision>` to `*_decomp.obj` (visual untouched). 15 decomp references
total (body0 + link2/4/5/7/hand/finger × L/R).
**Conservative tuning**: too-fine decomposition makes `getClosestPoints` 28× slower (see §5.3), so
link5/finger use 6 pieces and **link3 reverts to the original STL** (proximal, rarely collides, its
over-detection is harmless); total pieces ~309→101, slowdown down to 2.7×. True collisions are 100%
preserved (convex pieces never under-cover).

### 5.2 Blacklist (self-collision pair exclusions)
**Principle** (auto-decided by `gen_blacklist_openarm.py`): exclude only pairs with **no collision
signal** — permanent overlap (max distance < 0), rigidly-fixed (constant distance), parent-child
(URDF adjacent). **Keep any pair that can separate AND collide; never blacklist inter-arm pairs.**
**Pitfalls**: ① an early 56-pair list **wrongly blacklisted 20 pairs** (treating body0↔arm as phantom
and excluding all of it — wrong! ~39% are real collisions; fix the mesh, don't blacklist). Corrected
to **36 pairs** (current `blacklist_openarm.py`).
② **link5↔link7** (same arm, via link6): only 0.04% real collisions, but the coarse decomp still
slightly over-detects it → the sim stops on it spuriously; it is an intra-arm structural pair the
barrier cannot control → **kept blacklisted**.
③ **Intra-arm "skip-one" structural pairs (link1↔link3 etc., constant ~15mm, never collide)**:
leaving them unblacklisted does not cause false collisions (they are never < 0), but it **pollutes the
distance value** (pins min_dist at 15mm) → harms **regression / distance calibration** (see §5.6/§5.7).
`relabel_clean.py` excludes these 51 structural pairs (`structural_excl.json`) when relabeling.

### 5.3 Speed
- **OpenBLAS oversubscription**: the sim does many tiny linear-algebra ops per step (OSQP, inverse
  dynamics); OpenBLAS defaults to one thread per core → nearly all time spent on thread sync → **~8×
  slower**. Fix: set `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1` (scripts already setdefault).
- **`getClosestPoints` slows down with decomp**: convex decomposition explodes the pair count,
  O(pairs²) → full-pair distance query 28× slower; conservative decomp brings it to 2.7× (1.6× at a
  small threshold). **The physics step (`performCollisionDetection`) is only 1.3× slower, negligible.**
- **Real-time viewing**: the sim runs as fast as compute allows (no real-time sync). Add `--realtime`
  (sleep to wall-clock if ahead) and raise `--ctrl-dt` (e.g. 0.015) to cut per-step cost.

### 5.4 Acceleration / control period
- **`compute_qe` limit-clamping**: the braking-stop pose qe must be clamped to joint accel/position
  limits, otherwise the viability label is distorted.
- **Control-period bug**: the cls sim once mistook the 20ms CBF horizon for the control period →
  a 20ms ZOH let the swing-in torque coast and the arm self-collided → **contaminated all evaluations**.
  Fixed with **`--ctrl-dt 0.001` (1 kHz)** (the 20ms CBF/viability horizon is separate, `--box-dt`).
  **Always evaluate at ctrl-dt 0.001**; use 0.005 only for fast viewing.

### 5.5 OSQP solver
After migrating to Ubuntu, OSQP occasionally crashed → added a soft fallback (when the QP is
infeasible, eps is relaxed in steps down to 1e-3, then a greedy corner fallback); see the
`OSQP SolverError` branch in the sim.

### 5.6 Label poisoning (distance signal pinned by structural pairs)
The two gripper fingers (held at 0.01, a constant **14.5mm** gap) and similar zero-gradient structural
pairs, if not blacklisted, **dominate min_dist on 66% of configs**, pinning γ at 14.5mm and capping
the regressor's `grad_cos` at **0.45** (once misread as a capacity ceiling). Switching to a clean
"inter-arm distal distance" label (structural pairs excluded) + acceleration clamping raised
`grad_cos → 0.68` and the regressor avoided collisions.
**Lesson**: distance-value pollution only hurts regression / distance calibration, not binary
classification (structural pairs are never < 0, so they never flip the 0/1 label).

### 5.7 Using the classifier logit as a barrier: saturation, fluctuation, and the calibration trade-off
- **Symptom**: at a fixed dist (e.g. 1mm) the deployed γ fluctuates a lot (15→3); and the larger γ is,
  the more the barrier gradient saturates.
- **Root cause**: γ = logit margin; **CE constrains only the sign, not the magnitude** → in the safe
  region γ is free to fluctuate and saturate.
- **The attempted fix (make the logit margin represent distance)**: `loss = CE + 0.1·MSE(γ, 200·dist)`
  (`fast_train_margin.py`) → **fluctuation solved** (at a fixed dist, γ std 4.82→1.47; corr(γ,dist)
  0.88→0.979; acc 94.5% with zero loss; see `gamma_margin_d128.pt`).
- **But it revealed a counter-intuitive trade-off**: **γ-fluctuation and barrier-strength are coupled.**
  Calibrating γ to distance removes the fluctuation but also removes the barrier's punch → on the
  limit-cycle task (two overlapping circles force the arms very close), the calibrated model has **no
  usable operating point** (low threshold → collides; high threshold → brakes 100% of the time and
  tracks the circle ~150–377mm off). SCALE (redundant) / tanh (steep in the wrong place) / eps
  (ineffective) all failed to decouple them. **Conclusion: deploy the CE classifier; the γ fluctuation
  is cosmetic and does not hurt deployment.** Full record in `docs/MODELS_margin_calibration.md`.

### 5.8 DAgger backfire
Retraining the d128 classifier with "dangerous" configs collected from deployment trajectories raised
the collision rate **10%→80%**: the blind spot is "close-but-safe" (+1.9mm) configs; binary labels mark
them "safe" → the model overfits to "extremely safe / never intervene". The way out is **margin
classification** (`min_dist < margin`, not `< 0`), not feeding binary labels straight back.

### 5.9 The task's own ceiling
The classifier barrier was once stuck at ~80% success — the real culprit was not the model but the
**overlapping limit-cycle** (circle centers 6cm apart, radius 10cm) creating physically unavoidable
head-on configs. Moving the centers apart → 96.7%. **Confirm the task is feasible before blaming the
model** (the right arm's xy-circle is harder to reach, so radial tracking error is large for all
models — a task issue, not a model issue).

## 6. Mesh-fix details
V-HACD: `p.vhacd(in_obj, out_obj, log, resolution=, maxNumVerticesPerCH=, concavity=, minVolumePerCH=)`
(accepts only `.obj`; convert `.stl`→`.obj` with trimesh first). **Gotcha**: the arm meshes use
`meshScale = 0.001 -0.001 0.001` (Y-reflection; left/right fingers have opposite signs), so any
geometry computation must use the signed scale or it errs by ≤57mm. Validate with a probe: place a
sphere at a deep-concavity world point and check `getClosestPoints` clears it.
`scripts/visualize_convex_hulls.py` overlays hulls/decompositions for inspection.

## 7. Usable models (all `TransformerGamma`, d_model=128 / nhead=4 / num_layers=4, input 28-dim)

| Model | Type | Train data | Loss | Use / notes |
|---|---|---|---|---|
| `gamma_cls_v012_8M_d128.pt` | CE classifier | 8M v012 | CrossEntropy | **deployment default**; acc ~94.8% / recall ~95.5% |
| `gamma_cls_meshfix_d128.pt` | CE classifier | 8M (V-HACD geometry relabel) | CrossEntropy | geometry-fixed version; acc 94.6% / recall 94% |
| `gamma_margin_d128.pt` | margin calib. | 8M clean | CE + 0.1·MSE(γ,**200**·dist) | **solves γ fluctuation** (corr 0.979); but barrier is weaker (§5.7) |
| `gamma_margin_s500_d128.pt` | margin calib. | 8M clean | CE + 0.1·MSE(γ,**500**·dist) | SCALE study (conclusion: redundant knob) |
| `gamma_margin_tanh_d128.pt` | margin calib. | 8M clean | CE + 0.1·MSE(γ,10·**tanh**(dist/15mm)) | non-linear-target study (did not break through) |
| `gamma_reg_v012_8M_d128_linear.pt` | distance regression | 8M v012 | regression (linear head) | regresses min_dist directly; hurt by §5.6, barrier worse than the classifier |

**Deployment threshold**: CE models use `--gamma-threshold 8` (logit units); margin models use
calibrated units `γ = scale·dist` (the checkpoint stores `margin_scale`), so changing scale requires
re-tuning the threshold.

## 8. Training

```bash
# CE classifier (v012 recipe: 100ep + cosine + warmup3 + AdamW lr2e-4 + bs4096 + AMP, data on GPU)
python3 scripts/train/fast_train_cls.py output/openarm_dual_8M_clean.csv assets/models/out.pt 100
# margin distance-calibration (same recipe + MSE pinning γ to distance): trailing = <lam> <scale> [linear|tanh] [D0]
python3 scripts/train/fast_train_margin.py output/openarm_dual_8M_clean.csv assets/models/out.pt 100 0.1 200
```
Architectures tried: **CE classification** (baseline, most robust for deployment), **margin
calibration** (linear/tanh — solves fluctuation but weakens the barrier), **distance regression**
(linear head — hurt by label poisoning). **Conclusion: deploy the CE classifier.**

## 9. Key takeaways (to avoid repeating mistakes)
1. **Fix meshes before labeling**: PyBullet hull-izes concave meshes → ~25% phantom inter-arm labels;
   fix with V-HACD.
2. **Blacklist only no-signal pairs**: body0↔arm are real collisions (don't blacklist); intra-arm
   structural pairs hurt regression but not classification.
3. **Evaluate at ctrl-dt 0.001** and set OpenBLAS threads to 1, or it runs 8× slower.
4. **γ fluctuation is the price of a strong barrier**: calibrating it weakens control — deploy the CE
   classifier.
5. **Offline acc/recall is decoupled from closed-loop collision rate**: what actually moves the
   collision rate is the deployment threshold (sweep by collision rate, don't set it by recall).
6. **Confirm task feasibility first** (overlapping circles have physically unavoidable configs) before
   attributing failures to the model.

---
Built on VPP-TC (ICRA 2026). For the original method / single-arm Panda README, see the `main` branch.
