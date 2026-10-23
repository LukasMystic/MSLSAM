# 10 — MSLSAM v11 (Calibrated · Real-Data Protocol · Layer Ablation)

**Run:** 2026-09-15 14:11 · **Source:** `v11_calibrated.ipynb` · **Status:** ✅ **target met in Protocol B** — two prior conclusions overturned

## Headline

**ASR 4.18% at FPR 10.0%** (all-8 layers, Protocol B). The thesis target — ASR < 5%,
FPR < 10% — is met on the matched pair, in-domain.

Two caveats stated up front: the ASR confidence interval **[2.35, 7.33] straddles 5%**,
so n=263 cannot certify the result; and FPR sits *exactly* at the budget by construction,
not below it.

## Protocol A vs B — the decisive comparison

Both evaluated on DeepSet-Inj vs DeepSet-Benign, threshold at FPR=10%.

| | Protocol A (synthetic-trained) | Protocol B (real-trained, 5-fold CV) |
|---|---|---|
| Matched-pair AUC | 0.9149 | **0.9743** |
| TPR | 69.6% | **94.7%** |
| ASR | 30.4% | **5.3%** |
| Scores below 0.01 | 85.5% | **10.5%** |
| Threshold | 0.0004-ish scale | **0.3892** |

**B ≫ A.** Per the interpretation table written before the run: *the gap is
synthetic→real transfer, not a representational limit. The detector works when trained on
real data.*

This also answers `main.ipynb` cell [10], written months ago and never executed. Its
criterion was **mean AUC < 0.70 means hidden states do not encode the injection signal**.
Result: **0.9750 ± 0.0094**. The hidden states carry the signal clearly.

**Calibration was never a classifier bug.** Protocol A is still saturated (85.5% of
scores below 0.01); Protocol B is healthy (benign p50 = 0.040, injection p50 = 0.972,
threshold 0.389). Training on 40 synthetic prompts was *causing* the saturation. v9's
"collapse" and v10's Youden's J of 0.0014 were both symptoms of the same thing.

## Layer ablation — the L4 hypothesis is refuted

| Layer set | dim | OOF AUC | fold AUC | TPR | ASR | ASR 95% CI |
|---|---|---|---|---|---|---|
| **all8** | 40 | **0.9815** | 0.9824 ± 0.0076 | 95.82% | **4.18%** | [2.35, 7.33] |
| L4+8+12 | 15 | 0.9743 | 0.9750 ± 0.0094 | 94.68% | 5.32% | [3.20, 8.74] |
| **L12** | 5 | **0.9649** | 0.9651 ± 0.0161 | 93.54% | 6.46% | [4.07, 10.11] |
| L16+20+24 | 15 | 0.9517 | 0.9527 ± 0.0157 | 88.21% | 11.79% | [8.43, 16.25] |
| L8 | 5 | 0.9471 | 0.9469 ± 0.0211 | 85.55% | 14.45% | [10.71, 19.21] |
| L16 | 5 | 0.9390 | 0.9411 ± 0.0123 | 83.65% | 16.35% | [12.37, 21.30] |
| L24 | 5 | 0.9356 | 0.9376 ± 0.0298 | 84.41% | 15.59% | [11.70, 20.47] |
| L31 | 5 | 0.9317 | 0.9346 ± 0.0301 | 82.89% | 17.11% | [13.04, 22.13] |
| **L4** | 5 | **0.9176** | 0.9183 ± 0.0212 | 79.47% | 20.53% | [16.09, 25.82] |

**L4 is the worst single layer, not the best. L12 is the best.** Ranking:
L12 > L8 > L16 > L24 > L31 > L4.

### Why this contradicts the trajectory analysis
`_diagnostics/layer_trajectory.png` showed L4 carrying ~3× the margin of any other depth,
and that reading drove the v9/v10 early-layer design. The two measurements are not the
same thing:

| | Trajectory figure | v11 ablation |
|---|---|---|
| Negative class | WJB-Adv-Benign | DeepSet-Benign (matched) |
| Statistic | univariate mean-score margin | multivariate OOF AUC |
| Score | v8 Mahalanobis + refusal direction | trained prototype classifier |

They measure different quantities against different negatives. **For the matched-pair
task — which is the thesis task — v11 is the correct measurement and L4 loses.**

Within the combined L4+8+12 model, per-layer coefficient mass is nearly flat
(L12 5.32, L8 5.27, L4 4.56), so L4 still contributes; it is only weak in isolation.

## Operating curve (Protocol B, L4+8+12)

| FPR budget | threshold | TPR | ASR | ASR 95% CI |
|---|---|---|---|---|
| 20.0% | 0.2144 | 96.58% | 3.42% | [1.81, 6.37] |
| 15.0% | 0.2631 | 96.20% | 3.80% | [2.08, 6.86] |
| 12.5% | 0.3199 | 95.06% | 4.94% | [2.91, 8.27] |
| **10.0%** | 0.3892 | 94.68% | 5.32% | [3.20, 8.74] |
| 7.5% | 0.4445 | 92.40% | 7.60% | [4.98, 11.45] |
| 5.0% | 0.5938 | 88.59% | 11.41% | [8.11, 15.82] |
| 2.5% | 0.8179 | 77.57% | 22.43% | [17.81, 27.85] |

The target corner is right at the edge for L4+8+12; all-8 clears it.

## Prototype group importance
ContextPoison 3.66 · Authority 3.11 · RoleHijack 3.06 · Override 2.78 · **RealInj 2.53**

The synthetic category prototypes carry *more* weight than the real-injection prototypes.
The synthetic set fails as a training positive class but works well as a set of reference
directions. Top single feature: `P_ContextPoison_L12` (2.61).

## ⚠️ What this result does NOT show

**Protocol B's 10% FPR is measured against DeepSet-Benign only.** The 5-fold CV never
touches JBB-Benign (47% FPR in v10) or WJB-Adv-Benign (36%). The real-trained classifier
has not been tested against the safe sources that have driven false positives since
version 02. **This is the single biggest gap in the result.**

Also unresolved: in-domain 5-fold CV on one corpus is a ceiling, not a deployment
estimate — same corpus, same ~30% German mix, same attack style. Hard negatives (configs
C/D) still never ran. And the notebook again saved with outputs cleared; `results/v11/`
holds everything.

## Verdict
The question "is ASR < 5% and FPR < 10% possible with layer tracking on Llama?" now has
an evidence-backed answer: **yes, with in-domain training data.** The bottleneck was never
the representation, the classifier, or the layer choice — it was training positives drawn
from 40 synthetic prompts while evaluating on real ones.

## → v12
1. **Evaluate the real-trained classifier against the full safe suite.** Train on DeepSet
   folds, score JBB-Benign, WJB-Adv-Benign, XSTest-Safe, WJB-Vanilla-Benign. Until this
   runs, the FPR claim covers one benign corpus only.
2. **Save OOF scores per layer set**, not just the primary — all-8 is the best config and
   its per-prompt scores were not persisted.
3. **Expand the real injection corpus.** The ASR CI straddles 5% at n=263; only more data
   closes it. `injecagent.csv`, `tensortrust.csv`, `prompt_inject.csv` are still absent.
4. **Test cross-corpus generalization** — train on DeepSet, test on a held-out corpus.
   That converts a ceiling into a deployment claim.
5. Re-run the hard negatives (C/D) now that calibration is healthy.
