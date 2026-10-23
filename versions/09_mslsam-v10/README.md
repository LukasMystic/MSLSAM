# 09 — MSLSAM v10 (Prototype-Based)

**Run:** 2026-09-15 13:56 · **Source:** `v10_prototype.ipynb` (standalone) · **Status:** ⚠️ ranking fixed, calibration still broken, target missed

## Hypothesis
> The bottleneck is **direction quality, not classifier complexity.**

Replace v9's per-category centroid directions (estimated from ~10 examples) with
**max cosine similarity to any individual stored prototype**.

## Configuration
Layers **L4, 8, 12** · 4 categories × 10 prototypes each · PCA **99 components**
(capped by the 100-prompt baseline, not the configured 128) · LogisticRegression.

Ran **cache-only** — no model load. `v10_hard_neg_n20` was never extracted, so
**configs C and D did not run.**

| Config | Dim | Hard neg | Ran |
|---|---|---|---|
| A_ProtoOnly | 12 | no | ✅ |
| B_Proto+Cons | 15 | no | ✅ |
| C_Proto+HardNeg | 12 | yes | ❌ |
| D_Full | 15 | yes | ❌ |

## Headline results

| Metric | A_ProtoOnly | B_Proto+Cons | Target |
|---|---|---|---|
| Injection ROC-AUC | **0.9187** | 0.9177 | — |
| Injection AP | 0.7778 | 0.7763 | — |
| Global ROC-AUC | 0.8128 | 0.8140 | — |
| Youden's J threshold | **0.0014** | **0.0009** | — |
| FPR @ Youden | 20.08% | 20.48% | < 10% |
| ASR @ Youden | 14.03% | 14.03% | < 5% |
| ASR 95% CI | [10.4, 18.6] | [10.4, 18.6] | — |
| **TPR @ FPR=10%** | **73.38%** | 73.02% | ≥ 95% |
| **ASR @ FPR=10%** | **26.62%** | 26.98% | < 5% |

n(injection) = 278 · n(safe) = 1250

## ✅ What went right

**The prototype fix worked on ranking.** Injection AUC **0.831 → 0.919** against v9.
DeepSet detection at Youden is 85.2%, where v9 detected essentially nothing. The
hypothesis that unstable centroids were degrading v9 is supported.

**The detector is now injection-specific, correctly.** XSTest-Unsafe detection is
**0.6%** — harmful content that is not an injection is ignored. That is the right
behaviour, and it is why global AUC (0.813) is now *deflated* rather than inflated:
the inverse of the v8 problem. **Global AUC is no longer a meaningful number for this
detector and should not be reported as a headline.**

**Easy negatives are solved.** XSTest-Safe FPR **0.0%**, WJB-Vanilla-Benign **2.0%**.

## ❌ What went wrong

**1. The score collapse is NOT fixed.** Youden's J landed at 0.0014 — the same
pathology as v9 (0.001).

| Percentile | safe | injection |
|---|---|---|
| p50 | 2.8e-05 | 5.2e-02 |
| p90 | 5.8e-03 | 9.5e-01 |
| p99 | 1.2e-01 | 9.98e-01 |

**81.3% of all scores fall below 0.01.** The prototype change fixed *ordering* but not
*calibration* — LogisticRegression is still saturating. The probabilities are unusable
as an operating scale, so any threshold derived from them is arbitrary.

**2. The target is missed by a wide margin.** At the FPR=10% budget, ASR is **26.6%** —
more than five times the < 5% target. The ASR confidence interval [10.4, 18.6] sits
entirely above 5%, so this is a real miss, not a sample-size ambiguity.

**3. The matched pair is the actual failure — and the aggregate hides it.**

| Comparison | AUC | AP | TPR@FPR10 |
|---|---|---|---|
| Synthetic (n=15) vs all safe | 1.0000 | 1.0000 | **100.0%** |
| DeepSet-Inj vs easy safe only | 0.9900 | 0.9791 | 98.1% |
| DeepSet-Inj vs ALL safe | 0.9141 | 0.7542 | 71.9% |
| **DeepSet-Inj vs DeepSet-Benign** | **0.8624** | 0.8454 | **52.1%** |

The matched pair — same corpus, same language mix, injections vs non-injections — is
the real task. **At a 10% FP budget it catches barely half the injections.** The
headline 0.919 is an average over a solved problem (easy negatives, synthetic) and an
unsolved one.

**4. False positives are concentrated in three sources.**

| Safe source | FPR (A) |
|---|---|
| XSTest-Safe | 0.0% |
| WJB-Vanilla-Benign | 2.0% |
| DeepSet-Benign | **30.3%** |
| WJB-Adv-Benign | **35.7%** |
| JBB-Benign | **47.0%** |

**DeepSet-Benign at 30.3% is the important one.** It was never in the training negative
pool — negatives were `baseline` (100) + `sensitive_safe` (300) only. The classifier has
never seen a benign prompt from the corpus it is being evaluated against.

**5. F2 (category consensus) earns nothing.** A vs B across every metric: injection AUC
0.9187 vs 0.9177, FPR 20.08 vs 20.48, matched-pair AUC 0.8624 vs 0.8653. All within
noise. **Drop the 3 consensus features.**

**6. Notebook saved with outputs cleared**, so the per-config calibration AUC and the
`top feature` line were lost again. `results/v10/` caught the rest. The classifier
coefficients should be added to the JSON payload — without them we cannot tell which
layer is carrying the signal, which is the one thing the trajectory analysis says to
check.

## Verdict
The prototype hypothesis was **half right**. Direction quality was a real problem and
fixing it recovered the ranking. But v10 revealed the deeper issue: the detector
separates *injection-like phrasing from plain benign text*, not *injection from benign
within the same distribution*. That is the same confound the layer trajectory analysis
identified, now measured directly at 52.1% TPR@FPR10 on the matched pair.

## Artifacts
- `results/metrics.json`, `results/ablation.csv` (copies; live in `Experiments/results/v10/`)
- `figures/mslsamv10_{roc_curve,score_distribution,per_source,ablation}.png`
- `code/v10_prototype.ipynb` (snapshot)
- Full per-prompt scores: `Experiments/results/v10/scores_long.csv` (not copied, ~300 KB)

## → v11
Ordered by expected value, cheapest first.

1. **Fix calibration before touching architecture.** Inspect `clf.decision_function`
   (logits) rather than `predict_proba`; set operating points by score quantile; or fit
   isotonic/Platt calibration on a held-out split. The ranking is already decent — the
   scale is what is broken.
2. **Put DeepSet-Benign in the training negative pool.** 30.3% FPR on a corpus the
   model never saw during fit is the cheapest fixable error in the run.
3. **Run configs C and D.** Hard negatives target JBB-Benign and WJB-Adv-Benign
   directly — the two worst FP sources at 47% and 36%. One 20-prompt extraction.
4. **Drop F2.** Three features, no measurable contribution.
5. **Persist classifier coefficients** so the per-layer contribution is readable, then
   check whether L4 dominates as the trajectory analysis predicts.
6. **Report the matched pair as the headline metric.** DeepSet-Inj vs DeepSet-Benign is
   the honest task; everything else flatters the result.
