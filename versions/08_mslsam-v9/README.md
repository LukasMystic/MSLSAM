# 08 — MSLSAM v9 (Trajectory-Based)

**Run:** 2026-06-16 11:57 · **Source:** `main.ipynb` cell [7] · **Status:** ❌ score collapse

## Design
Pure latent-space monitor — no Constitutional Judge, no secondary LLM. Layers **L4, 8, 12, 16**
(early, on the hypothesis that injection signals spike early and benign prompts evolve smoothly).

**26 features in 4 families:**
| Family | Dim | Content |
|---|---|---|
| F1 — Multi-directional injection projections | 16 | 4 semantic categories × 4 early layers. **A separate direction per category**, not a single attack centroid. |
| F2 — Layer trajectory consistency | 6 | 3 cosine similarities + 3 L2 norms between consecutive layer pairs |
| F3 — Trajectory curvature | 3 | Second-order cosine changes + total curvature |
| F4 — Early-vs-late shift ratio | 1 | Injection spikes early; benign evolves smoothly |

Single `LogisticRegression` per ablation config:
`A_Projection` (16d, F1) · `B_Trajectory` (6d, F2) · `C_Proj+Traj` (22d, F1+F2) ·
`D_All` (26d, all four, primary).

## Results
| Metric | Value |
|---|---|
| ROC-AUC | 0.831 |
| Youden's J threshold | **0.001** |
| Detection rate | 70.6% (1033 / 1464) |
| FPR | 24.7% (309 / 1250) |

Confusion matrix at t=0.001: TN 941 · FP 309 · FN 431 · TP 1033.

## What actually went wrong
**The score distribution collapsed.** In the per-source boxplot, every source except
Synthetic-Injections (≈1.0) and DeepSet-Injections sits flat against zero — including
JBB-Attacks and XSTest-Unsafe, which earlier versions detected at 95–98%. The optimal
threshold falling to 0.001 is a probability-compression signature: the classifier outputs
near-zero for almost everything.

**The 0.831 AUC is an artifact.** It is carried almost entirely by ~65 synthetic injections
that are trivially separable. Strip them and there is little left.

## Likely cause (untested)
Not a representational ceiling. The layer trajectory analysis — same hidden states, same
layers — shows a clean, stable injection/benign gap at every depth. The signal is there.

The suspect is **direction quality**: F1 estimates 4 separate category directions from a
calibration pool of only ~10–15 synthetic examples each. Noisy directions → near-orthogonal
projections → a degenerate logistic fit. This is exactly the hypothesis v10 was written to
test, and v10 was never run.

## Ablation
The A/B/C/D ablation results were not saved. Re-running would show whether F2–F4 (the
trajectory families proper) contribute anything over F1 alone — directly relevant to the
thesis claim, since trajectory tracking is the stated method.
