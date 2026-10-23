# 06 — MSLSAM v7 (Dual-Signal MoE)

**Run:** 2026-06-16 · **Source:** `main.ipynb` cell [6], §6 `TrajectoryClassifier` · **Status:** ⚙️ intermediate — no separate figures

## Design
Replaces the hand-weighted score blend with a learned classifier over a **16-dimensional
feature vector: 8 Mahalanobis + 8 refusal-direction values**, one per hooked layer
(L4, 8, 12, 16, 20, 24, 28, 31).

The design point: **the two signals are kept separate, not blended.** The forest learns
per-layer, per-signal importance independently, exploiting both geometry (Mahalanobis) and
direction (projection) rather than collapsing them into one number as versions 02–05 did.

### Mixture of experts
Two independent RandomForests:
- `inj_clf` — trained on injection examples (Type A + Type B)
- `harm_clf` — trained on harmful-content examples (HarmBench / JBB / WJB / XSTest)

Inference: `score = max(p_inj, p_harm)` — flagged by either expert. Both share one
`RobustScaler` fitted on the union of training data.

```python
RandomForestClassifier(
    n_estimators=100, max_depth=4, min_samples_leaf=5,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
```

Negatives for `inj_clf` = baseline + sensitive-safe.

## Results
**Not recorded separately.** v7 and v8 share cell [6]; only the final state of that cell
saved figures, and those are attributed to v8 (version 07). The isolated contribution of
the MoE structure is therefore unmeasured.

## Verdict
Architecturally the most interesting idea in the classical line — the 16-dim unblended
feature vector is what cells 10 and 11 later audit as "the v8 representation", and
`max_depth=4` with 16 features is a deliberately low-capacity model, which is the right
choice at this sample size.

**Open question this version should have answered and did not:** does the MoE actually beat
a single classifier trained on both pools? An ablation with `inj_clf` alone, `harm_clf`
alone, and one merged forest would be cheap on the cached representations and would tell
you whether `max(p_inj, p_harm)` is earning its complexity.
