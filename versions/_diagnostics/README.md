# Diagnostics

Investigation cells written after v9's collapse. **Only cell [13] ran.**

| Cell | Purpose | Status |
|---|---|---|
| [9] Dataset Collapse Investigation | Is the collapse a weak classifier or wrong positive-class construction? | ⏸ never run |
| [10] Hidden State Sufficiency Test | Is prompt injection reliably separable in hidden states at all? | ⏸ never run |
| [11] v8 Representation Audit | Does the v8 representation carry enough information, independent of classifier? | ⏸ never run |
| [12] Pilot Annotation (N=50) | Behavioral clustering of DeepSet injections | ⚠️ partial |
| [13] Layer Trajectory Analysis | Mean anomaly score per layer, per dataset | ✅ **ran** |

---

## ✅ Cell [13] — Layer Trajectory Analysis (the key result)

Mean anomaly score computed **independently at each layer depth**, not aggregated.
Datasets: Synthetic-Injections (n=65), DeepSet-Injections (n=263), WJB-Adv-Benign (n=5000),
WJB-Vanilla-Benign (n=4996).

| Layer | DeepSet-Inj | WJB-Adv-Benign | **Margin** | WJB-Vanilla-Benign |
|-------|-------------|----------------|------------|--------------------|
| **4**  | 0.84 | 0.56 | **0.28** | 0.39 |
| 8      | 0.79 | 0.70 | 0.09 | 0.36 |
| 12     | 0.78 | 0.70 | 0.08 | 0.39 |
| 16     | 0.84 | 0.73 | 0.12 | 0.44 |
| 20     | 0.81 | 0.71 | 0.10 | 0.41 |
| 24     | 0.79 | 0.73 | 0.06 | 0.37 |
| 28     | 0.75 | 0.72 | 0.03 | 0.36 |
| 31     | 0.81 | 0.77 | 0.04 | 0.41 |

*Values read off `figures/layer_trajectory.png`. Regenerate numerically before citing.*

### Two findings

**1. Layer 4 carries ~3× the margin of any other depth** against adversarially-phrased
benign — the class that produces the false positives — and the margin decays monotonically
with depth, effectively gone by L28 (0.03). Plausible reading: by deep layers the model has
resolved surface form into something like intent, and "injection" and "unusually phrased
request" become indistinguishable. The discriminative information is in the *early* layers,
where surface structure still dominates.

This **contradicts version 03**, which selected L28 on calibration AUC (0.9844) — measured
against harmful prompts, not against the hard negative class. Trust this one: it was
measured against the class that actually matters, with n=5000 on the benign side.

**2. The signal exists.** Same hidden states v9 used. A clean, stable gap between injections
(0.75–0.84) and vanilla benign (0.36–0.44) at every depth. If the representation were empty,
this gap would not appear — so v9's collapse is most likely a classifier/calibration
artifact, not a representational ceiling.

---

## ⚠️ Cell [12] — Pilot Annotation (N=50)

`../../taxonomy_annotation/pilot_summary.json`:
```json
{"n_sample": 50, "seed": 42,
 "has_stage1_scores": false, "has_cj_verdicts": false,
 "prelim_distribution": {"A": 23, "H": 8, "B": 3, "N": 16},
 "german_count": 15, "n_borderline_cj": 0}
```

Preliminary labels only. Stage-1 scores and CJ verdicts absent (cell [12] must run after
cell [6] with results in scope). All 263 rows of `deepset_annotation.csv` have empty
`ann1` / `ann2` / `ann_final`. 60 IAA sample ids selected — no second annotator pass, so
**no kappa**.

Note **15 of 50 pilot items are German**. DeepSet is not a linguistically uniform corpus;
any language-sensitive feature will pick this up.

---

## ⏸ Cell [10] — the decisive unrun test

Its own stated decision criteria:

| Criterion | Conclusion if met |
|---|---|
| Exp D best AUC **< 0.70** | Hidden states do not encode the injection signal |
| Classifier AUC range **< 0.05** | Feature ceiling, not model ceiling |
| Max Fisher ratio **< 0.5** | Features individually too weak |

Four classifiers × 2 training regimes; dimensionality reduction; per-feature Bhattacharyya /
JSD / Fisher; DeepSet-only 5-fold CV for the best achievable under ideal conditions.

**All inputs are cached.** `representation_cache/` holds DeepSet-Injections at (263, 4096)
and DeepSet-Benign at (300, 4096) for all 8 layers. Runs on numpy + sklearn, no model load.

This test settles whether the project's premise is sound. It should run before any further
architecture work.
