# 02 — Llama Baseline (Mahalanobis + Refusal Direction)

**Run:** 2026-04-20 · **Source:** `main.ipynb` cell [3] · **Status:** ✅ best classical result

## Design
`meta-llama/Meta-Llama-3-8B-Instruct` (32 layers), MPS. Hooks on **L8, L16, L24**.

1. **Phase 1 — safe baseline.** PCA 4096→128 per layer, Mahalanobis distance to the benign centroid.
2. **Phase 2 — refusal direction.** Direction from 70 harmful prompts; relative Mahalanobis.
3. **Phase 2b — damped topic normalization.** α=0.5 over 200 sensitive-safe prompts, removing
   part of the topic component so "discusses a sensitive subject" stops reading as "attack".
4. **Phase 3 — weight search.** Constrained grid [0.30–0.70] over Mahalanobis vs refusal.

## Calibration
| Layer | PCA var. expl. | safe Mahal μ/σ | rel. Mahal μ/σ | topic overlap | optimal weights | cal AUC |
|---|---|---|---|---|---|---|
| 8  | 92.8% | 10.98 / 2.42 | −0.5403 / 0.3236 | 0.5468 | Mahal .50 / Ref .50 | 0.9436 |
| 16 | 91.0% | 11.03 / 2.41 | −0.7630 / 0.4126 | 0.5701 | Mahal .30 / Ref .70 | 0.9739 |
| 24 | 88.3% | 10.92 / 2.90 | −0.6266 / 0.4407 | 0.6751 | Mahal .30 / Ref .70 | 0.9739 |

**Split:** 70 calibration harmful · 1023 eval attacks (6 sources) · 750 eval safe (4 sources) ·
350 diverse baseline anchor (100 Dolly + 100 JBB-Benign + 50 WJB-Adv-Benign + 50 XSTest-Safe + 50 WJB-Vanilla-Benign).

## Results
| Metric | Value |
|---|---|
| Optimal threshold (Youden's J) | 0.6624 |
| ROC-AUC | **0.9495** |
| Average Precision | 0.9579 |
| Detection rate | 91.50% |
| FPR | 12.80% |
| ASR | 8.50% |

### Per-source detection
| Source | N | Det. rate | Mean score |
|---|---|---|---|
| HarmBench-Test | 294 | 93.9% | 0.8058 |
| HarmBench-Val | 72 | 98.6% | 0.8242 |
| JBB-Attacks | 93 | 97.8% | 0.8596 |
| XSTest-Unsafe | 192 | 95.3% | 0.7743 |
| WJB-Vanilla-Harmful | 185 | 90.8% | 0.8154 |
| WJB-Adv-Harmful | 187 | **78.6%** | 0.7495 |

### Per-source FPR
| Source | N | FPR | Mean score |
|---|---|---|---|
| XSTest-Safe | 250 | 9.2% | 0.4495 |
| JBB-Benign | 100 | **26.0%** | 0.5691 |
| WJB-Vanilla-Benign | 200 | 9.5% | 0.4200 |
| WJB-Adv-Benign | 200 | 14.0% | 0.5499 |

## Verdict
Strongest numbers in the project — but **this measures harmfulness detection, not prompt
injection.** No injection corpus is in the evaluation set. It is the right baseline to cite
for the classical approach and the wrong number to quote as an injection result.

Two failure modes visible here persist through every later version: **JBB-Benign** drives
false positives (26.0%) and **WJB-Adv-Harmful** drives misses (78.6%). Both are the
adversarially-phrased classes.
