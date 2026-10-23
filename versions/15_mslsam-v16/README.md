# 15 — MSLSAM v16 (Hard-Negative Training)

**Status:** ✅ complete — hard-negative training is a **trade, not a fix**. First defensible headline. · **Source:** `v16a_join_caches.ipynb` → `v16b_analysis.ipynb`

**No new GPU time.** Both notebooks run off caches that already exist.

## Why v16

v15's premise was falsified — an injected payload is not a localised disturbance of the layer
trajectory, and position-resolved features lose to eight whole-document scalars. But v15 left
behind the control no earlier version had: **tier-2 hard negatives**, a benign imperative
sentence inserted into an untrusted document at the same position, length-matched to the payload
it replaces.

Against those, v14's detector fires at **40.75–52.39%**, versus 19–20% on the untouched carrier
and 60–71% on real injections. v15's finding (b):

> the detector is substantially detecting that **an instruction is present** in untrusted
> content, not that the instruction is **malicious**.

**But it was never shown a benign instruction during training.** That is the whole of v16: is
finding (b) a property of the representation, or an artefact of the training set?

## Why it costs nothing to run

| source | provides | rows |
|---|---|---|
| `representation_cache/unified` (v14) | last-token states — direct families, all 8 benign corpora, Dolly baseline | 9,910 + 2,000 |
| `representation_cache/chunked_v15` | **final chunk = the last token** — indirect families **and the hard negatives** | 3,335 |

The non-indirect rows are byte-identical across the two corpora (**9,910/9,910 exact text
match**), and v15's final chunk closes on the last real token by construction, so it *is* v14's
feature. v16a asserts that both caches share model, system prompt, `max_len` and layer set rather
than assuming it — a mismatch would silently encode "which cache did this row come from".

## The grouping problem, and why there are two keys

The new leak: an injected row and its hard-negative twin **share a carrier**. If one is in train
and the other in test, the model can recognise the *document* — which would make arm B win for
entirely the wrong reason.

The obvious fix, the connected component of the (`payload_id`, `pair_id`) graph, is unusable
where it matters most:

| contrast | components | largest |
|---|---|---|
| BIPIA + BIPIA-HardNeg | 51 | 317 rows (18.9%) |
| InjecAgent + InjecAgent-HardNeg | **1** | **695 rows (100%)** |

InjecAgent has 17 templates and 62 attacker instructions, each appearing across several
templates, so its graph is fully connected. Under component grouping **every InjecAgent row lands
in one fold**, which silently turns Exp 1 into leave-one-family-out for InjecAgent and makes its
numbers incomparable with v14's. This project has been bitten by that class of invisible
question-change before, so both keys are emitted and the notebook says which it used:

- **`group_carrier`** (primary) — `pair_id` for indirect rows, `payload_id` elsewhere. Closes the
  new twin leak; keeps the *same* payload-across-carriers leakage profile v14 had, so v16 stays
  comparable with v14.
- **`group_component`** (sensitivity) — stricter, used only where it yields enough groups.
  Exp 3 reports both and treats the gap between them as the uncertainty.

## The four experiments

| # | Question | Decides |
|---|---|---|
| 1 | **Arm A vs Arm B** — hard negatives held out of training vs included | does showing the boundary teach it, and what does it cost elsewhere? |
| 2 | **Nested per-context thresholds** | the honest headline ASR, properly nested at last |
| 3 | **Injection vs hard negative, directly** | can the representation separate them *at all*? |
| 4 | **Cost and complexity** | owed since 2026-09-27 |

**Exp 3 is the scientific centre.** Both classes carry an imperative sentence inside untrusted
content, same position, length-matched; only one is an attack. Read it like this:

- best AUC **< 0.70** → no classifier separates them. Finding (b) is a property of the latent
  space, not of v14's training set. That is a strong negative claim and a thesis result.
- best AUC **> 0.85** → separable when asked directly; v14's detector simply never was asked,
  and hard-negative training is the fix.
- in between, or grouping-sensitive → report the **range**, not a point estimate.

Exp 3 sweeps all four classifiers, because v14's lesson was that the head is what mattered.

## Two corrections carried in

- **v14's ASR 7.09% was optimistic** — its per-context thresholds were fitted on the rows they
  were measured on. Exp 2 nests them (calibration half, measurement half, 400 resamples). This is
  the first defensible version of that number.
- **v15's Exp 2 was mis-specified** — the chunk cache held only two benign contexts, so its
  "threshold spread collapsed" was an artefact of dropping the corpora that created the spread.
  v16 has the full eight-corpus suite, so the question is answerable here.

## Method notes
- v14's exact feature set, unchanged: prototype max-cosine per family × 8 layers + Mahalanobis
  × 8 layers = 24 features. Self-match masking removes the self **column** before the max — not
  compared against it (v11 got this wrong positionally, v15c by comparison).
- Nested thresholds, Wilson CIs, per-corpus FPR printed beside every pooled number.
- `nested()` returns `nan` for an empty extra slice rather than crashing — arm A has no
  hard-negative rows in its OOF by construction.
- Verified before release: both notebooks compile, an AST use-before-define pass is clean across
  all 13 code cells, and a synthetic dry-run confirms fold feasibility, the empty-slice path, and
  that the two arms differ in the intended direction.

## What would falsify / complicate this
- Arm B drops hard-negative FPR **but** raises ASR or the other benign FPRs → the boundary is
  learnable only by trading away detection; report the trade, not a win.
- Arm B drops hard-negative FPR **only** under `group_carrier` and not `group_component` → it
  learned the carrier, not the boundary.
- Exp 3 AUC high under `group_carrier` and low under `group_component` → payload memorisation,
  not generalisation.

## v16a run (2026-09-28) — the join is sound

13,245 rows: **9,910 from the v14 cache + 3,335 from the v15 chunk cache**, 8,042 carrier groups,
7,596 component groups, 1.7 GB of layer files.

### Commensurability gate: PASS

State-norm ratio, v15-sourced ÷ v14-sourced:

| layer | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 31 |
|---|---|---|---|---|---|---|---|---|
| ratio | 0.918 | 0.883 | 0.875 | 0.953 | 0.984 | 0.952 | 0.963 | 0.963 |

All inside the 0.5–2.0 band, so the two halves are on the same scale.

### The ~5% offset is the text, not the cache — checked, not assumed

The v15 half sits consistently ~5% lower. If that were a cache artefact it would be a shortcut
feature, because source correlates with family (everything indirect comes from v15). So it was
tested against the v14 cache **alone**, where there is only one source and any gap must be the
text:

| layer 16, v14 cache only | median norm |
|---|---|
| indirect corpora (BIPIA, InjecAgent, their clean) | **8.25** |
| everything else | **8.70** |

The same gap. Long retrieved documents simply produce a slightly smaller last-token state. It
cannot encode "which cache", and Exp 3 is unaffected in any case since both its classes come from
the v15 half.

## v16b run 1 crashed — stale column reference

`AttributeError: 'DataFrame' object has no attribute 'group'`. Two leftover `df.group` /
`ev.group` references from before the key was split into `group_carrier` / `group_component`.
Fixed, and a regex scan confirms none remain.

**Fourth bug of this shape in two versions.** The AST use-before-define pass cannot catch it:
`df.group` is attribute access on a DataFrame, not a bare name, so it compiles cleanly and fails
at runtime. The rule now in AGENTS.md: after renaming a DataFrame column, grep for the old name
both as `.name` and as `["name"]` across every notebook.

## Results (2026-09-28)

11,245 evaluation rows, 6,042 carrier groups, PCA 512, HistGradientBoost, `group_carrier`.

### Exp 3 — the pre-registered centre: *partially* separable

The reading rule was fixed before the run. The answer landed in the middle band.

| contrast | grouping | best AUC | best ASR @ 10% FPR |
|---|---|---|---|
| BIPIA vs BIPIA-HardNeg | `group_carrier` | **0.8256** (RF) | 47.44% |
| BIPIA vs BIPIA-HardNeg | `group_component` | **0.7984** (MLP) | 50.35% |
| InjecAgent vs InjecAgent-HardNeg | `group_carrier` | **0.8116** (RF) | 51.25% |

So the boundary between *a malicious instruction* and *a benign one, same position, same length,
same document* is **real but weak**: AUC 0.80–0.83, ASR 45–58% at a 10% budget.

That refutes the strongest form of v15's finding (b) — the representation is not blind to
maliciousness — while confirming its substance: the signal is far too weak to act on. The
stricter component grouping costs BIPIA only −0.027 AUC, so the conclusion is robust to the
grouping choice rather than an artefact of it.

### Exp 1 — arm B is a trade, and the pre-registered falsifier fired

| | arm A (held out) | arm B (trained) |
|---|---|---|
| ROC-AUC | **0.9820** | 0.9726 |
| ASR % | **5.30** [4.66, 6.03] | 8.27 [7.47, 9.15] |
| pooled FPR % | 10.02 | 10.01 |

| benign corpus | A FPR % | B FPR % | Δ |
|---|---|---|---|
| **BIPIA-HardNeg** | **79.26** | **57.40** | −21.87 |
| InjecAgent-Clean | 58.51 | 9.74 | −48.78 |
| BIPIA-Clean | 35.95 | 34.24 | −1.71 |
| **InjecAgent-HardNeg** | 22.03 | 8.08 | −13.95 |

| attack corpus | A det % | B det % | Δ |
|---|---|---|---|
| BIPIA | 98.00 | 93.75 | −4.25 |
| **InjecAgent** | **76.88** | **29.57** | **−47.31** |
| SPML | 96.32 | 96.76 | +0.44 |
| DeepSet | 76.05 | 77.95 | +1.90 |

Training on hard negatives does lower hard-negative FPR — by 22 points on BIPIA and 14 on
InjecAgent. But **InjecAgent detection collapses from 76.9% to 29.6%**, AUC falls, and ASR rises
from 5.30% to 8.27%. Arm B did not learn the instruction/injection boundary; it learned *not to
fire on tool-output-shaped content at all*, which drops its false positives and its true
positives together.

This is exactly the falsifier written into this README before the run: **a trade, not a win.**
And BIPIA-HardNeg still sits at 57.40% after training on that very class.

### Exp 2 — the first defensible headline, and v14's 7.09% was optimistic

Properly nested per-context thresholds (calibration half, measurement half, 400 resamples):

| family | threshold | ASR % [Wilson] | own-context FPR % |
|---|---|---|---|
| **SPML** | 0.006127 | **0.91 [0.61, 1.38]** | 9.91 |
| DeepSet | 0.011848 | 8.07 [5.28, 11.90] | 10.76 |
| BIPIA | 0.971377 | 31.26 [28.69, 33.93] | 10.52 |
| InjecAgent | 0.702137 | 47.06 [40.26, 54.47] | 10.10 |
| **POOLED** | — | **12.21** | — |

**12.21%, not 7.09%.** Two things were wrong with v14's number and both are now fixed:
its thresholds were fitted on the rows they were measured on, and **its InjecAgent FPR was
estimated from 17 negative rows**. v16 has 479. The 19.35% v14 reported for InjecAgent was
meaningless at that n; the real figure is 47.06%.

Arm B is worse again at 14.78%, consistent with Exp 1.

### The threshold spread is real and did not collapse

Arm A: **5.43e3×** across 8 benign contexts (0.000179 XSTest → 0.973114 BIPIA-Clean), against
v14's 7.5e3×. v15's apparent collapse was an artefact of having only two contexts in the chunk
cache, exactly as suspected. **Calibration across contexts remains an open, unsolved problem.**

### Exp 4 — cost, finally measured

| | per prompt |
|---|---|
| feature construction (24 features, 8 layers, 512 PCA dims, 4,148 prototypes) | **3.212 ms** |
| classifier inference | **0.005 ms** |
| **detector total** | **3.22 ms** |
| Llama-3-8B forward pass (v15b, measured on this machine) | ~2,000 ms |
| **detector overhead** | **+0.16%** |

Asymptotics per prompt (n tokens, d = 4096, k = 512, L = 8 layers, P = 4,148 prototypes):

| component | cost |
|---|---|
| forward pass | O(n²d + nd²) — dominates; one prefill |
| PCA projection | O(Ldk) |
| prototype max-cosine | **O(LkP)** — linear in the training set |
| Mahalanobis | O(Lk²) |
| HistGradientBoost | O(trees × depth), effectively constant |

The honest framing: **the guard costs one prefill of the prompt it is protecting, plus 3 ms.** An
LLM-judge baseline costs a full generation. The one thing that will not scale is prototype
max-cosine, which is linear in P — cap or cluster the prototypes before a production corpus.

## What the thesis can now claim

**On direct prompt injection the target is met.** SPML, n = 2,499, nested threshold, own-context
FPR 9.91%: **ASR 0.91% [0.61, 1.38]**. Comfortably inside ASR < 5% and FPR < 10%.

**On indirect injection it is not**, and the reasons are now named rather than guessed:
1. carrier contexts need thresholds ~5,400× apart, and nothing tried has closed that;
2. the instruction-vs-injection boundary is only weakly encoded (AUC 0.80–0.83);
3. training on the boundary trades detection away rather than learning it.

**Required caveat on the SPML claim:** its injections and its benign rows come from the same
source corpus, so this is an *in-distribution* result. v14's leave-one-corpus-out put SPML at AUC
0.72 held out. State it as "direct injection, in-distribution", never as a general claim.

## Status

v16 is complete. Hard-negative training is not the fix. The thesis result is a **scoped positive**
(direct injection, in-distribution, ASR 0.91%) plus three characterised failure modes for
indirect injection — which is a more defensible contribution than a single pooled number would
have been. Fill in after v16b, then update `EXPERIMENTS.md` and `AGENTS.md` (R2/R3/R5).

| Metric | v14 run 2 | v15 | v16 arm A | v16 arm B | Target |
|---|---|---|---|---|---|
| Context-matched ASR % (nested) | 7.09 (not nested) | — | | | < 5% |
| Worst benign FPR % | 47.33 | — | | | < 10% |
| Hard-negative FPR % | — | 40.75–52.39 | | | report |
| Exp 3 best AUC | — | — | | | see rule above |
| Threshold spread | 7.5e3× (8 ctx) | mis-specified | | | — |
