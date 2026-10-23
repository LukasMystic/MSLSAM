# 11 — MSLSAM v12 (Honest Operating Points)

**Run:** 2026-09-15 15:54 · **Source:** `v12_honest.ipynb` · **Status:** ✅ ran · in-domain claim survives, **generalisation claims do not**

## Headline

**Best config: all-8 layers, DeepSet-only negatives, C=10, grouped CV → ASR 3.20%,
AUC 0.9847, FPR 10.34%.** The best result in the project, and the first one measured
without leakage in the folds or the threshold.

Two different uncertainties apply and they are not the same thing:

| Config | ASR | threshold/fold variation | binomial (n=263) |
|---|---|---|---|
| **all8 / deepset / C=10** | **3.20%** | [2.66, 4.18] | **[1.55, 5.89]** |
| all8 / deepset / C=1 | 4.08% | [3.42, 5.83] | [2.35, 7.33] |
| L4+8+12 / deepset / C=10 | 4.76% | [3.42, 6.34] | [2.91, 8.27] |

The threshold-variation interval clears 5%; the **binomial interval does not**. Only the
second one licenses the claim "ASR < 5%", so the target is **still not certified** — but
the shortfall is now small:

| At ASR | injections needed | shortfall |
|---|---|---|
| **3.20%** | **368** | **105** |
| 4.08% | 1,777 | 1,514 |
| 4.26% | 2,862 | 2,599 |

**~105 more real injections closes it.** v11's position implied 2,000+.

## Experiment 1 — leakage was real but modest

| Config | folds | AUC | ASR % [95%] |
|---|---|---|---|
| L4+8+12 | random | 0.9769 | 5.27 [4.11, 7.00] |
| L4+8+12 | **group** | 0.9701 | **7.10 [5.40, 9.28]** |
| all8 | random | 0.9824 | 3.42 [2.89, 4.26] |
| all8 | **group** | 0.9782 | **4.26 [3.27, 5.93]** |

Near-duplicate leakage inflated AUC by 0.004–0.007 and understated ASR by **0.8–1.8
points**. Smaller than feared, but real, and it runs the wrong way for a thesis claim.
All numbers from here on use grouped folds.

## Experiment 2 — cross-source FPR is NOT solved

| Safe source | t from DeepSet-Benign | t from mixed pool |
|---|---|---|
| XSTest-Safe | 0.01% | 0.35% |
| WJB-Vanilla-Benign | 0.56% | 1.21% |
| JBB-Benign | 4.04% | **11.23%** |
| DeepSet-Benign | **10.19%** | **14.91%** |
| **WJB-Adv-Benign** | **11.83%** | **21.84%** |
| ASR at that threshold | 13.11% | 9.57% |

**WJB-Adv-Benign breaches the budget under both calibrations** — 11.8% at best, 21.8% at
worst. Adversarially-phrased benign text has defeated every version since 02 and still
does. JBB-Benign is fine (4%) when calibrated on DeepSet but blows to 11.2% on a mixed
pool.

⚠️ **Confound:** this experiment ran with `neg_pool = deepset+base`, which Experiment 4
subsequently showed is the *worse* pool by ~9 ASR points. The cross-source FPR numbers
therefore come from a weakened model and should be re-measured with
`all8 / deepset / C=10`. Direction of the finding is unlikely to change; magnitude may.

## Experiment 3 — cross-language transfer degrades badly

| | AUC | ASR % [95%] | n train / test |
|---|---|---|---|
| In-language (grouped CV) | 0.9782 | **4.26** | — |
| EN → DE | 0.9522 | **10.35** [6.80, 18.45] | 160 / 103 |
| DE → EN | 0.8829 | **35.01** [28.75, 44.38] | 103 / 160 |

Training on English and testing on German roughly **doubles ASR**. The reverse direction
**collapses** — ASR 35%. Part of the asymmetry is training-set size (160 vs 103), but not
all of it: AUC drops 0.095 as well.

**This means the detector is substantially lexical.** It is not keying purely on
instruction-override structure; a language it has not been trained on largely defeats it.
That directly limits what the thesis can claim.

## Experiment 4 — the negative pool matters more than anything else

| C | negatives: DeepSet only | + baseline/sensitive_safe | delta |
|---|---|---|---|
| 0.1 | 5.70% | 14.41% | **+8.72** |
| 1.0 | 4.08% | 13.12% | **+9.04** |
| 10.0 | **3.20%** | 11.37% | **+8.18** |

**Adding `baseline` and `sensitive_safe` to the training negatives costs ~9 points of
ASR**, consistently across regularisation. Those pools (Dolly, XSTest-Safe, JBB-Benign,
WJB benign) pull the decision boundary away from the DeepSet distribution. This was
recommended after v10 on the reasoning that the classifier should see the corpus it is
judged against — the data says the opposite, decisively.

Weaker regularisation also helps monotonically (C=10 > C=1 > C=0.1), consistent with the
feature space being low-dimensional and clean rather than noisy.

## Threshold instability confirmed
FPR 95% range is **[4.4, 18.1]** across every config — a threshold estimated from 150
benign prompts is simply not a reportable quantity. See
`figures/mslsamv12_threshold_instability.png`.

## Verdict

| Claim | Status |
|---|---|
| In-domain, same-language detection at ASR ~3.2% | ✅ holds, leakage-corrected |
| ASR < 5% certified | ❌ not yet — needs ~105 more injections |
| FPR < 10% across benign corpora | ❌ WJB-Adv-Benign breaches at 11.8–21.8% |
| Generalises across languages | ❌ ASR 10–35% cross-language |

The defensible thesis claim is now narrower and much better evidenced:

> *Latent-space prototype detection over Llama-3-8B hidden states reaches ASR 3.20% at a
> 10% FPR budget on in-domain, in-language prompt injection (grouped CV, n=263,
> binomial 95% CI [1.55, 5.89]). Cross-language transfer degrades to 10–35% ASR, and
> adversarially-phrased benign text remains above the FPR budget at 11.8%.*

## → v13
1. **Re-run Experiment 2 with `all8 / deepset / C=10`.** The cross-source FPR numbers are
   currently measured on the weaker pool.
2. **Collect ~105–150 more real injections.** That is the whole remaining gap on
   certification. `injecagent.csv`, `tensortrust.csv`, `prompt_inject.csv` are still absent.
3. **Attack WJB-Adv-Benign directly** — it is the only safe source over budget. The v10
   hard negatives (never extracted) target exactly this; §5 of `v10_prototype.ipynb`.
4. **Decide the language claim.** Either train bilingually and claim in-language only, or
   investigate what makes DE→EN collapse. Do not claim language generality.
5. Sweep C beyond 10 — the trend has not turned.
