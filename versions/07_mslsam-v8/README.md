# 07 — MSLSAM v8 (Policy Thresholds)

**Run:** 2026-06-16 11:36 · **Source:** `main.ipynb` cell [6] (final state) · **Status:** ⚠️ headline number is the wrong metric

## ⚠️ Figure filename warning
This version's figures are saved as **`mslsamv3_*.png`** — a stale `save_path` from an
earlier edit. They are v8 output; the print header directly above the save calls reads
`MSLSAM v8 EVALUATION RESULTS`. Copies in `figures/` keep the original names. **Do not cite
them as "v3" in the thesis.**

## Design
v7's dual-expert MoE plus threshold policy. Introduces **injection-stratified evaluation**:
thresholds calibrated on injection-only scores (Type A = Synthetic + DeepSet), not on the
harmfulness-dominated pooled set.

Three operating points reported:
| Policy | Rule |
|---|---|
| Stratified Youden's J | maximize TPR − FPR |
| Deployment policy | FPR ≤ 9.5% |
| Security policy | **ASR ≤ 5.0%** ← the thesis target |

## Results
| Metric | Value |
|---|---|
| Global ROC-AUC (all attacks vs safe) | **0.944** |
| Injection-only ROC-AUC (Type A) | **NOT RECOVERED** |
| Per-policy FPR / Detection / ASR | **NOT RECOVERED** |

The per-source boxplot shows clean separation: all benign sets pinned below ~0.05;
HarmBench / JBB / XSTest-Unsafe / WJB-Harmful medians above 0.78. **DeepSet-Injections —
the only real injection corpus — is the weakest attack class**, median ≈0.51 with a range
straddling the threshold.

## Verdict
**The 0.944 is not an injection result.** It is the global AUC, computed over an attack set
dominated by HarmBench, JBB-Attacks and XSTest-Unsafe — harmful-intent prompts, not
injections. The code computes the honest number (`injection_auc`) in the same run and
prints it to stdout, but cell [6] has no saved outputs, so it is lost.

Recovering it requires only a re-run against `representation_cache/` — no model load. This
is the single highest-value cheap experiment available, because it is the number v9 should
have been compared against. The apparent 0.944 → 0.831 "regression" at v9 may not be a
regression at all.

## Action
Re-run cell [6] and capture the `TARGETED EVALUATION: PROMPT INJECTION (Type A Only)` block.
Fix the `save_path` strings to `mslsamv8_*` first.
