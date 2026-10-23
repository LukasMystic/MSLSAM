# 13 — MSLSAM v14 (Unified Multi-Corpus)

**Status:** ✅ run 2 complete (HistGradientBoost) · **Source:** `v14a_build_corpus.ipynb` → `v14b_extract.ipynb` → `v14c_analysis.ipynb`

First version trained on more than one injection corpus, and the first with **indirect**
injection at all.

## Why v14

v13 ended with no configuration meeting ASR < 5% and FPR < 10%: either ASR 3.10% with
99.7% FPR on WJB-Adv-Benign, or all benign corpora under budget at ASR 11.78%. The binding
constraint was separating injection from adversarially-phrased benign, on 263 injections
from a single corpus of mostly direct override attacks.

v14 attacks that with data and protocol, not architecture.

## Three-notebook pipeline

| Notebook | Needs model | Output |
|---|---|---|
| **v14a** build corpus | no | `datasets/unified/injection_corpus.csv` + manifest |
| **v14b** extract | **yes** | `representation_cache/unified/shard_*.npz` (checkpointed, resumable) |
| **v14c** analysis | no | `results/v14/metrics.json`, figures |

## Corpus design

| Source | Unique payloads | Family |
|---|---|---|
| SPML | 12,541 (840 system-prompt groups) | direct, persona override |
| DeepSet | 263 (172 clusters) | direct, EN+DE |
| BIPIA | **250** | **indirect** — email / code / table |
| InjecAgent | **62** | **indirect** — tool output |
| ~~HackAPrompt~~ | excluded | 71.4% contain "PWNED" — competition target string, not an injection feature |

Three design decisions that carry the version:

**`payload_id` is the CV group key, not text similarity.** BIPIA's 250 payloads expand
combinatorially to tens of thousands of rows; 4-gram Jaccard cannot cluster the same
payload across two carrier documents because the carrier dominates the n-grams. Grouping
by text would leak worse than v11 did.

**Matched clean carriers.** Every BIPIA/InjecAgent injected example emits the *same*
carrier with no payload as a negative, sharing `carrier_id`. The detector must find the
payload, not recognise the document — the most direct attack available on the confound.

**PCA baseline raised 100 → 2,000.** v13's PCA was capped at 99 components by a
100-prompt baseline; v14 allows up to 512.

## The four experiments

| # | Question | Decides |
|---|---|---|
| 1 | Payload-grouped CV, full corpus | the new headline, measured against every benign corpus at once |
| 2 | Leave-one-corpus-out | does it generalise across attack families, or learn corpus fingerprints? |
| 3 | Length-matched control | v13's matched pair gave AUC 0.793 from character count alone — is that dead? |
| 4 | Separability ceiling | can ANY classifier separate injections from WJB-Adv-Benign? |

Exp 4's decision rule: classifier spread < 0.05 means a **feature** ceiling (more capacity
will not help — change the representation); best AUC < 0.70 means the features genuinely
do not carry the signal.

## Method notes to carry into the thesis
- BIPIA/InjecAgent deliver payloads via retrieved content or tool output; the monitor
  formats `system + user`, so carrier+payload is placed in the user turn. Standard
  simplification — state it rather than letting an examiner find it.
- Self-match masking is **identity-based**: a query row that is itself a prototype has that
  one similarity masked. Positional masking (v11–v13) is only correct when prototypes are a
  prefix of the query set, which stopped being true once prototypes became a scattered
  subset. Fixed here.
- Operating points stay nested — threshold on a calibration half, measured on a held-out
  half, resampled — with both the threshold-variation interval and the binomial CI.

## Results — run 1 (2026-09-27, LogisticRegression)

Corpus: 9,838 rows · 4,148 injections / 5,690 benign · 6,548 payload groups · PCA 512 · C = 10.

> **Reporting bug, now fixed:** `metrics.json` from run 1 records `pca_components: 190`. That is
> wrong. The variable holding the PCA dimension was rebound by Exp 3's length-matching loop
> before the persist cell read it; the actual PCA dimension in every experiment was **512**.
> Renamed to `N_PCA` in the patched notebook. Cite 512, not 190.

| Metric | Exp 1 (full) | Exp 3 (length-matched) | Target |
|---|---|---|---|
| ROC-AUC | 0.9051 | 0.8823 | — |
| AP | 0.8759 | 0.8768 | — |
| ASR % [thr-var] | 28.15 [27.12, 29.63] | — | **< 5%** |
| ASR % [Wilson] | [26.81, 29.55] | — | **< 5%** |
| FPR % (pooled) | 10.05 | 9.97 | < 10% |
| Worst benign corpus | **BIPIA-Clean 43.96%** | — | < 10% |
| Exp 2 worst held-out | **InjecAgent AUC 0.5534 / ASR 94.11%** | — | — |
| Exp 4 best AUC / spread | **0.9958 (HGB) / 0.0549** | — | — |

### Exp 1 — payload-grouped CV, full corpus
Per-corpus FPR at the pooled 10% operating point:

| Benign corpus | FPR % |
|---|---|
| **BIPIA-Clean** | **43.96** ← over budget |
| **InjecAgent-Clean** | **11.66** ← over budget |
| **WJB-Adv-Benign** | **10.24** ← over budget |
| WJB-Vanilla-Benign | 4.71 |
| XSTest-Safe | 3.28 |
| JBB-Benign | 3.00 |
| SPML-Benign | 2.58 |
| DeepSet-Benign | 0.78 |

Detection rate by injection corpus: SPML 77.8% · BIPIA 67.9% · InjecAgent 50.0% · DeepSet 49.8%.

The BIPIA-Clean number is the headline failure and it is the *informative* one: those are the
**matched clean carriers** — the same email/code/table documents with the payload removed. A
44% false-positive rate on them means the linear detector is firing on the carrier document,
not on the payload. The matched-carrier control worked exactly as designed: it caught a
confound that every previous version would have scored as success.

### Exp 2 — leave-one-corpus-out

| Held out | AUC | AP | ASR % |
|---|---|---|---|
| BIPIA | 0.5486 | 0.268 | 74.09 |
| DeepSet | 0.6325 | 0.073 | 80.90 |
| InjecAgent | 0.5534 | 0.032 | 94.11 |
| SPML | 0.7191 | 0.436 | 81.06 |

Near-chance on both indirect families. Under LogisticRegression the detector learns corpus
fingerprints, not a family-invariant injection signature. This is the single most damaging
result in the version — and the one most likely to change under run 2, since it was measured
with the classifier Exp 4 shows to be the bottleneck.

### Exp 3 — length-matched control ✅
AUC 0.8823 matched vs 0.9051 unmatched: **−0.0228**. v13's length shortcut (AUC 0.793 from
character count alone) is effectively dead. The corpus rebuild fixed it. One objection
retired.

### Exp 4 — separability ceiling (injections vs WJB-Adv-Benign, n = 4,148 / 2,500)

| Classifier | ROC-AUC | ASR % | ASR % [Wilson] | FPR % |
|---|---|---|---|---|
| LogisticRegression | 0.9409 | 14.02 | — | 10.16 |
| RandomForest | 0.9942 | 1.06 | — | 10.10 |
| **HistGradientBoost** | **0.9958** | **0.62** | **[0.43, 0.92]** | **10.05** |
| MLP | 0.9939 | 1.09 | — | 10.01 |

**Spread 0.0549 — above the 0.05 decision threshold.** Under the rule stated above, this is
*not* a feature ceiling. The features carry the signal; the linear head could not read it.

## The finding

Every version from v09 to v14 used LogisticRegression. On the exact class that has blocked the
project since v02 — adversarially-phrased benign — swapping the head for a gradient-boosted
tree moves ASR from 14.02% to **0.62% at FPR 10.05%**, inside the target on both axes.

The signal was in the trajectory features all along. The bottleneck was the decision boundary
being linear in PCA space, which the ablations never varied because the classifier was never
treated as a hyperparameter.

**Consequence:** Exp 1's ASR 28.15% and Exp 2's LOCO collapse are measurements of
LogisticRegression, not of the representation. Neither licenses a conclusion until re-run.

## Results — run 2 (2026-09-27, HistGradientBoost)

`PRIMARY_CLF = "HistGradientBoost"`, PCA 512 (correctly reported now), C = 10, same 9,838 rows.

### Exp 1 — both heads, same folds

| | LogisticRegression | **HistGradientBoost** |
|---|---|---|
| ROC-AUC | 0.9048 | **0.9867** |
| AP | 0.8749 | **0.9825** |
| ASR % | 28.35 | **3.25** [2.76, 3.84 Wilson] |
| Pooled FPR % | 10.05 | 9.99 |

Detection rate: BIPIA 66.6 → **96.7** · DeepSet 48.7 → **86.3** · InjecAgent 50.0 → **100.0** ·
SPML 78.2 → **97.6**.

**ASR 3.25% at pooled FPR 9.99% is inside the target on both axes — and it is not a
deliverable result.** The pooled FPR hides the per-corpus breakdown:

| Benign corpus | n | HGB FPR % | LogReg FPR % |
|---|---|---|---|
| **InjecAgent-Clean** | 17 | **100.00** | 0.07 |
| **BIPIA-Clean** | 525 | **47.33** | 45.11 |
| **JBB-Benign** | 100 | **14.84** | 3.00 |
| WJB-Adv-Benign | 2,500 | **7.49** ✅ | 10.16 |
| DeepSet-Benign | 300 | 5.45 | 1.09 |
| SPML-Benign | 1,198 | 5.23 | 2.64 |
| WJB-Vanilla-Benign | 800 | 2.67 | 4.28 |
| XSTest-Safe | 250 | 0.48 | 3.20 |

**The v02 confound is solved.** WJB-Adv-Benign — adversarially-phrased benign, the class that
has blocked this project since version 02 and that v13 named as the binding constraint — is at
**7.49%, under budget**, while detection rose across every attack family. That is the result
v14 was built to get.

**A new binding constraint replaced it: the indirect-injection carriers.** BIPIA-Clean at 47%
contributes ~248 of the pooled false positives, roughly 4.4 of the 9.99 points. InjecAgent-Clean
is 17/17. JBB-Benign regressed 3.00 → 14.84.

### Exp 2 — leave-one-corpus-out: improved, still failing

| Held out | LogReg AUC | **HGB AUC** | Δ | HGB ASR % |
|---|---|---|---|---|
| BIPIA | 0.5486 | **0.7031** | +0.155 | 82.73 |
| DeepSet | 0.6325 | **0.8256** | +0.193 | 58.66 |
| InjecAgent | 0.5534 | **0.8042** | +0.251 | 76.72 |
| SPML | 0.7191 | **0.7790** | +0.060 | 62.81 |

The predicted risk did **not** materialise — the tree did not memorise corpus fingerprints
harder than the linear head. Every family improved, two of them by ~0.2 AUC. But ASR 59–83%
held-out means cross-family generalisation remains unsolved, and this is now the honest
statement of the method's limit.

### Exp 3 — length-matched (HGB)
AUC 0.9730 vs 0.9867 unmatched: **−0.0137** (was −0.0228 under LogReg). ASR 7.42% at FPR 10.25%
on the balanced n = 2,012 / 2,012 subset. Length shortcut confirmed dead.

### Exp 4 — ceiling, unchanged and stable
HGB 0.9958 / ASR 0.61% · RF 0.9942 / 1.11% · MLP 0.9937 / 1.15% · LogReg 0.9410 / 13.96%.
Reproduces run 1 to within 0.0001 AUC.

---

## Post-hoc: is it the payload or the document? (`code/v14d_posthoc_threshold.py`)

The 100% and 47% clean-carrier FPRs look like the detector classifying *documents* rather than
finding *payloads*. Pairing each injected row against its own clean carrier says otherwise:

| | matched pairs | mean score injected | mean score clean | paired margin (median) | injected > its own clean |
|---|---|---|---|---|---|
| BIPIA | 1,200 / 1,200 | 0.8721 | 0.2259 | **+0.8807** | **92.2%** |
| InjecAgent | 186 / 186 | 0.9818 | 0.7853 | **+0.0696** | **89.2%** |

The detector *does* find the payload — 89–92% of injected documents outscore the identical
document with the payload removed. **The failure is threshold placement, not detection.**

Threshold each benign context would need for its own 10% FPR:

| Context | local 90th pct |
|---|---|
| InjecAgent-Clean | 0.994230 |
| BIPIA-Clean | 0.799264 |
| JBB-Benign | 0.065324 |
| WJB-Adv-Benign | 0.016514 |
| DeepSet-Benign | 0.009106 |
| SPML-Benign | 0.008205 |
| WJB-Vanilla-Benign | 0.001946 |
| XSTest-Safe | 0.000132 |

**Nearly four orders of magnitude**, against a single global threshold of 0.0416. Tool-output
and retrieved-document contexts sit intrinsically high; short clean instructions sit at zero.
No single scalar cutoff can serve that spread.

### The honest operating point

Hold **every** benign context to 10% FPR and score each injection against its own context:

| Family | threshold | ASR % | misses |
|---|---|---|---|
| SPML | 0.008205 | **1.24** | 31 / 2,499 |
| DeepSet | 0.009106 | **7.98** | 21 / 263 |
| BIPIA | 0.799264 | **17.17** | 206 / 1,200 |
| InjecAgent | 0.994230 | **19.35** | 36 / 186 |
| **Pooled** | — | **7.09** | **294 / 4,148** |

**ASR 7.09% is the number to defend, not 3.25%.** It is the best honest result the project has
produced (v13's FPR-compliant point was 11.78%) and it is 2.1 points from target. The entire
remaining gap is the two indirect families; the direct families are already inside budget.

*Caveat: these thresholds are fitted on the same benign rows they are measured on. v15 must
nest them (calibration half / measurement half, resampled) before this number is cited.*

## Bug found in v14a — fix before v15

`carrier_id` is hashed **with the corpus name in it**, so an injected row gets
`BIPIA:<hash>` and its clean twin `BIPIA-Clean:<hash>` — different keys for the same document.
The matched-carrier pairs exist in the data but **cannot be joined by key**; the analysis above
had to recover them by text-prefix matching. Emit the *same* `carrier_id` for both members of a
pair in v14a.

## v15 direction

1. **Context-relative scoring.** Score the untrusted content's context baseline separately and
   threshold on the *margin*, not the absolute score. In real indirect-injection deployment the
   retrieved document / tool output is separable from the rest of the prompt, so this is a
   deployable design, not a leak.
2. **Nest the per-context thresholds** and re-measure the 7.09%.
3. **Fix `carrier_id`** and make the paired margin a first-class reported metric — it is the
   only metric that distinguishes payload detection from document recognition.
4. **InjecAgent n = 17 clean carriers is too small to conclude from.** Expand it or drop it from
   the FPR table and report it as a qualitative probe.
5. Explain the JBB-Benign regression (3.00 → 14.84) before it appears in a thesis table.

---

## Superseded: run 1 notes (LogisticRegression)

### Run 2 patch, as planned (now executed)

`v14c_analysis.ipynb` patched:
- `N_PCA` replaces the shadowed `k` (fixes the 190/512 reporting bug)
- classifier registry (`make_logreg` / `make_hgb` / `make_rf`), `PRIMARY_CLF = "HistGradientBoost"`
- Exp 1 runs **both** LogReg and HGB into `E1_ALL`, so run 1 stays comparable
- Exp 2 and Exp 3 now use HistGradientBoost

Open question for run 2: HGB's win in Exp 4 is *within* one corpus pairing. Exp 2 asks whether
it also generalises across families. A tree with 300 boosting rounds has far more capacity to
memorise a corpus fingerprint than a linear model does, so LOCO could get **worse**, not better.
That is the result to read first.
