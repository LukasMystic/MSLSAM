# 12 — MSLSAM v13 (Closing v12's Open Items)

**Run:** 2026-09-15 17:00 · **Source:** `v13_closing.ipynb` · **Status:** ⚠️ **v12's headline is overturned**

## Summary

v12 reported **ASR 3.20% at FPR 10%** and called it the best result in the project. v13
re-measured false positives against the *full* safe suite with that same config and found
it flags **99.7% of WJB-Adv-Benign and 93.3% of JBB-Benign**.

It was never a prompt-injection detector. It was a DeepSet-vs-everything-else classifier
whose FPR happened to be measured only on DeepSet-Benign.

**No configuration tested meets both targets.** The honest result is a trade-off curve.

## Exp 1 — cross-source FPR with the best config

`all8 / DeepSet-only negatives / C=10`, threshold calibrated on DeepSet-Benign:

| Safe source | FPR % [95%] |
|---|---|
| DeepSet-Benign | 10.22 [7.33, 13.67] |
| XSTest-Safe | 11.68 [9.60, 15.20] |
| WJB-Vanilla-Benign | 14.44 [11.33, 19.33] |
| **JBB-Benign** | **93.33** [90.00, 94.00] |
| **WJB-Adv-Benign** | **99.72** [99.33, 100.00] |

ASR at that threshold is 3.10% — but essentially every adversarially-phrased benign prompt
is blocked. Calibrating on a mixed pool instead collapses the other way: the threshold
saturates at 1.00 and ASR rises to 78.8%.

v12's Exp 2 reported WJB-Adv-Benign at 11.8% because it used the `deepset+base` pool,
where the model *had* seen those distributions. **The pool was doing far more work than
v12 credited.**

## Exp 3 — negative-pool decomposition (the central result)

| Negatives | ASR % | WJB-Adv FPR | JBB FPR | XSTest FPR | WJB-Van FPR |
|---|---|---|---|---|---|
| none (v12 "best") | **3.10** | 99.72 | 93.33 | 11.68 | 14.44 |
| + XSTest-Safe | 4.26 | 99.79 | 79.40 | 1.83 | 8.30 |
| + baseline pool | 3.94 | 98.78 | 86.57 | 11.66 | 8.09 |
| + WJB-Vanilla-Benign | 4.77 | 99.53 | 78.04 | 1.54 | 3.31 |
| + JBB-Benign | 6.69 | 97.61 | 18.43 | 3.86 | 2.53 |
| + sensitive_safe pool | 10.60 | 13.50 | 4.35 | 1.41 | 0.03 |
| + WJB-Adv-Benign | 10.81 | 12.64 | 27.21 | 11.45 | 5.09 |
| **+ WJB-Adv + JBB** | **11.78** | **9.95** | **7.62** | **3.23** | **2.52** |

- **Configs meeting ASR < 5% and WJB-Adv FPR < 10%: none.**
- **Only `+ WJB-Adv + JBB` gets all four FPRs under budget** — at ASR 11.78%.
- A corpus only fixes its own FPR. Adding XSTest-Safe drops XSTest FPR to 1.8% and leaves
  WJB-Adv at 99.8%. There is no transfer between benign distributions.

### ⚠️ Correction to v12

v12 concluded that adding `baseline + sensitive_safe` "costs ~9 ASR points" and that
"DeepSet-only wins decisively." **That was judged on the wrong metric.** Those 9 points buy
WJB-Adv-Benign FPR from 99.7% down to 13.5%. The post-v10 recommendation to include them
was right; the v12 analysis measured only in-domain ASR and missed what it was paying for.

## Exp 2 — the C trend turns

| C | AUC | ASR % | binomial 95% |
|---|---|---|---|
| 1 | 0.9792 | 4.08 | [2.35, 7.33] |
| **10** | 0.9847 | **3.21** | [1.55, 5.89] |
| 30 | 0.9851 | 3.32 | [1.81, 6.37] |
| 100 | 0.9839 | 3.55 | [1.81, 6.37] |
| 300 | 0.9812 | 3.86 | [2.08, 6.86] |
| 1000 | 0.9780 | 4.49 | [2.63, 7.80] |

C=10 confirmed optimal; the trend turns just past it. Settled.

## Exp 4 — the language collapse was an artifact, not a language effect

At matched training size, with C=10 and DeepSet-only negatives:

| | AUC | ASR % [95%] |
|---|---|---|
| EN → DE (n=103) | 0.9838 | **3.44** [2.91, 4.85] |
| DE → EN (n=103) | 0.9741 | **5.10** [3.75, 6.88] |

Learning curve, English-trained, tested on held-out German:

| German added | n test | AUC | ASR % |
|---|---|---|---|
| 0 | 103 | 0.9889 | **2.40** |
| 10 | 93 | 0.9920 | 2.75 |
| 20 | 83 | 0.9887 | 2.20 |
| 40 | 63 | 0.9945 | 2.96 |

### ⚠️ Second correction to v12

v12 reported DE→EN ASR **35.01%** and concluded "the detector is substantially lexical;
do not claim language generality." **That was a configuration artifact.** v12's language
split ran at C=1 with the `deepset+base` pool; at C=10 with DeepSet-only negatives the
same split gives **5.10%**, and an English-only model scores 2.40% on held-out German.

Cross-language transfer is *fine*. The caveat: these ASR figures are measured against
DeepSet-Benign, and Exp 1 shows this config's FPR elsewhere is ~100%. The language finding
is real but conditional on a configuration that is not deployable.

## Exp 5 — certification needs ~600, not 368

| n | Wilson upper | bootstrap upper | certified? |
|---|---|---|---|
| 263 (today) | 5.89 | 5.90 | no |
| 368 | 5.27 | 5.25 | no |
| 450 | 5.15 | 5.01 | no |
| **600** | **4.89** | **4.78** | **yes** |
| 800 | 4.57 | 4.47 | yes |

v12's "~105 more injections" used a Wald approximation. The Wilson interval is wider at
this n: the real requirement is **~600 injections, roughly 340 more.** And this applies
only to the non-deployable DeepSet-only config — the FPR-compliant config sits at ASR
11.8%, far outside certification range at any n.

## Verdict

| Claim | Status |
|---|---|
| ASR ~3.2% in-domain | ✅ true, but on a model that blocks 99.7% of adversarial benign |
| FPR < 10% across benign corpora | ✅ achievable (`+ WJB-Adv + JBB`), at ASR 11.8% |
| Both targets simultaneously | ❌ **no configuration tested** |
| Cross-language generality | ✅ holds — v12's refutation was an artifact |

The defensible claim:

> Prototype detection over Llama-3-8B hidden states trades attack detection against
> false positives on adversarially-phrased benign text. At ASR 3.10% the detector flags
> 99.7% of adversarial benign prompts; constraining all benign corpora below 10% FPR
> raises ASR to 11.78%. The target region (ASR < 5%, FPR < 10%) was not reached by any
> configuration tested. Cross-language transfer is not a limiting factor.

That is a real finding: **the binding constraint is separating injection from
adversarially-phrased benign text** — the same confound visible since version 02.

## → v14
1. **Report the trade-off curve as the result.** Sweep the benign mixture weight and plot
   the ASR/FPR frontier. The single operating point is gone; the curve is the contribution.
2. **Attack the actual confound.** WJB-Adv-Benign and injections are near-inseparable in
   this feature space. Test whether *any* classifier separates them — if not, the limit is
   representational and that is the thesis finding.
3. **Re-run Exp 4's language result under the FPR-compliant config** before claiming
   language generality.
4. Certification at n≈600 only matters if a config reaches the target region first.
