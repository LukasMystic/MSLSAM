# 05 — IDP-CJ (Injection-Direction Projection + Constitutional Judge)

**Run:** 2026-06-14 · **Source:** `main.ipynb` cell [6] · **Status:** ⚠️ hard ceiling found

## The pivot
**This is where the project changes target.** Versions 01–04 detect *harmful content*.
From here on the target is *prompt injection* — instruction-hierarchy override. Numbers
before and after this line are not comparable.

## Design
| ID | Change |
|---|---|
| IDP-1 | **Phase-2 calibration swapped**: harmful datasets → synthetic injection examples. The direction vector now points toward instruction-override intent, not harmful content. `compute_baseline()` API and body unchanged — only the data. |
| IDP-2 | **ConstitutionalJudge**: the same frozen Llama-3-8B-Instruct in `generate()` mode. Receives **only the user message**, never the deployment system prompt, to prevent circular evaluation. Runs on borderline inputs only — ~25–30% overhead. |
| IDP-3 | Combined score `max(stage1, cj_score)`. Stage 1 catches representation-shift injections; CJ catches text-level attempts not yet reflected in activations. |
| IDP-4 | `SYNTHETIC_INJECTION_EVAL` held out from Phase-2 calibration to prevent threshold leakage. |

**Calibration set:** 50 synthetic Type-A injections in 5 categories — direct override (10),
role hijacking (10), authority spoofing (10), delimiter injection (5), indirect/context
poisoning (5) — plus 10 Type-B (benign task-switch: *"Forget the previous text. What is the
capital of France?"*). `topic_damping=0.75`. Sensitive-safe pool = JBB-Benign + XSTest-Safe
+ WJB-Adv-Benign.

The code also tries to load `datasets/injecagent.csv`, `datasets/tensortrust.csv` and
`datasets/prompt_inject.csv` — **all three are absent and silently skipped.**

## Results
| Metric | Value |
|---|---|
| ROC-AUC | **0.871** |

## Verdict
The ROC curve **plateaus at TPR ≈ 0.88 from FPR 0.15 all the way to 0.46.** That flat
segment means roughly 12% of attacks are unreachable at *any* threshold — a representational
or feature problem, not a tuning problem. No amount of threshold search recovers them.

This finding motivated abandoning the single-direction approach for the multi-category
projections of v9.

Adding the Constitutional Judge did not lift AUC above the classical detector. Versions
v9 and v10 drop it entirely ("No Constitutional Judge. No secondary LLM.") — worth noting
that the CJ was removed on cost/purity grounds, not because it was shown to be useless.
