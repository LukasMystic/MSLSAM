# 14 — MSLSAM v15 (Position-Relative Detection)

**Status:** ❌ complete — **premise falsified.** Two findings worth keeping · **Source:** `v15a_build_corpus.ipynb` → `v15b_extract_chunked.ipynb` → `v15c_analysis.ipynb`

First version to track the layer trajectory across **position as well as depth**, and the first
with negatives hard enough to falsify the claim.

## Why v15

v14 run 2 changed the picture twice. It **solved** the confound that had blocked the project
since v02 — WJB-Adv-Benign FPR 10.16% → **7.49%**, under budget, with detection up on every
family. And it **exposed** a new one: the honest operating point, with every benign context
held to 10% FPR, is **ASR 7.09%**, and the whole remaining gap is the two indirect families
(BIPIA 17.17%, InjecAgent 19.35%).

The post-hoc analysis said exactly what is wrong. Paired against its own clean carrier, the
injected document scores higher in **92.2%** (BIPIA) and **89.2%** (InjecAgent) of pairs — so
the detector *finds the payload*. What it cannot do is share a threshold: the cutoff each
context needs for its own 10% FPR spans

| Context | local 10%-FPR threshold |
|---|---|
| InjecAgent-Clean | 0.994230 |
| BIPIA-Clean | 0.799264 |
| XSTest-Safe | 0.000132 |

**Nearly four orders of magnitude, against one global value of 0.0416.** Ordering within a
context is good; calibration across contexts is broken. v15 fixes calibration.

## The idea

v14b took one hidden state per row — `h[:, -1, :]`, the last token. That vector carries the
entire document, so a tool-output carrier and a one-line question land in different regions of
the space and cannot share a cutoff.

v15b takes the hidden state at the **end of every 32-token chunk**, out of the *same forward
pass*. The states were always there; v14b discarded them.

Because the model is causal, the state at chunk *k* encodes chunks 1..*k*. So the **first
difference** `D_k = Z_k − Z_(k−1)` isolates what chunk *k* contributed. A statistic like
`max(D) − median(D)` then asks:

> does **one part** of this document behave unlike the rest of it?

That is scale-free with respect to the document's baseline elevation — the exact quantity that
spanned four orders of magnitude. A uniformly-elevated benign tool output has a small spike; an
injected one has a large spike, wherever in the document it sits.

It is also the thesis's own idea extended by one axis: **v14 tracks trajectory across depth,
v15 tracks it across depth and position.**

## Three changes in v15a

**1. The `carrier_id` bug is fixed.** v14a prefixed it with the corpus name, so an injected row
got `BIPIA:<h>` and its clean twin `BIPIA-Clean:<h>` — different keys for the same document,
and the matched pairs could not be joined at all. The post-hoc analysis had to recover them by
text-prefix matching. Both members now share `pair_id`.

**2. Two tiers of negative, never pooled.**

| Tier | Construction | A false positive here is |
|---|---|---|
| `*-Clean` | payload slot removed, or declarative filler | unambiguously wrong |
| `*-HardNeg` | same carrier, same position, **benign imperative** filler | defensible — report separately |

Tier 2 is new and it is the real test. A Dolly instruction dropped into a retrieved document
is imperative, task-shaped, and benign. If the detector fires on it, the detector is finding
*instruction-shaped text in untrusted content*, not *an instruction that overrides the
assistant's task*. That distinction is the thesis claim and nothing before v15 could measure it.

Arguably flagging tier 2 is correct behaviour under a strict trust boundary — which is why it
is reported separately and never folded into the headline FPR.

**3. InjecAgent negatives: 17 → ~510.** v14 run 2 reported "100% FPR" on InjecAgent-Clean. That
was **17 rows**, all sharing one constant filler string. An FPR on 17 rows is not an FPR. The
dataset has only 17 tool-response templates — a hard ceiling — but each has exactly one
`<Attacker Instruction>` slot, so both tiers scale by varying the *filler* instead.

**4. Payload character spans recorded**, so localisation can be checked against ground truth.

## The four experiments

| # | Question | Decides |
|---|---|---|
| 1 | **Ablation** — v14 last-token vs v15 position vs both | does position help at all, same folds, same classifier, same negatives |
| 2 | **Threshold comparability** | does the four-order-of-magnitude spread collapse? **this is the actual fix** |
| 3 | **Localisation** | does the flagged chunk contain the payload? |
| 4 | **Indirect LOCO** | train BIPIA → test InjecAgent and back |

Exp 2 is the one that matters: v15 is a calibration fix, so the headline is whether one
threshold now serves contexts that needed 7,500× different cutoffs.

Exp 3 is the one a better classifier cannot manufacture. If the argmax chunk lands on the
payload, the detector is doing what the thesis says it does — and it is a figure for the
professor, not just a number. It is scored against a **majority-class baseline**, not uniform
chance: BIPIA places the payload at start or end 50/50, so a constant guess already beats
uniform without localising anything.

## Method notes
- PCA is refitted **in chunk space** on Dolly chunk states, not reused from v14's last-token
  basis. A mid-document state is not distributed like an end-of-prompt state.
- Prototypes are the delta vector **at the payload's own chunk** of each training injection —
  supervised at training time only. Inference never sees a span.
- Self-match masking removes the whole self column *before* the max, not after. Comparing
  against the max would be v11's positional bug in a new costume.
- Tier-2 hard negatives are never trained on. They are scored by every fold model and averaged,
  so the threshold and the scores come from the same models (v12's Exp 2 mixing bug).
- Nested thresholds, grouped folds on `payload_id`, Wilson CIs — unchanged from v12/v14.
- The numerical core was dry-run on synthetic arrays before release: feature shapes, NaN
  handling for short documents, fold coverage, and the Exp 2/3 paths all execute.

## Cost

v15b is the same number of forward passes as v14b — the extra cost is disk, not time. Only the
indirect families plus a 400-row Dolly baseline are extracted (~3,900 rows, **~2 h** at v14b's
measured 0.5 rows/s, ~2 GB at fp16). The direct families already meet target (SPML 1.24%,
DeepSet 7.98%) and are not re-extracted.

If 2 h is too long, lower `BIPIA_PER_PAYLOAD` in v15a and rebuild. **Do not lower `MAX_LEN` in
v15b** — it would change the chunk grid and break comparability with v14.

## What would falsify this

- Exp 2 spread stays above ~100×  → position features do not fix calibration; the fix has to
  come from the representation, and v15's premise is wrong.
- Exp 3 exact ≈ majority baseline → no localisation; `max−median` is picking up something
  other than the payload, and Exp 1 gains (if any) are not what they look like.
- Tier-2 FPR ≈ tier-1 FPR → the detector cannot tell a benign instruction in untrusted content
  from an injection, which is a genuine limit worth reporting rather than hiding.

## v15a run 1 (2026-09-27) — and why it is being re-run

13,025 rows. Spans located in **100%** of injected rows. `pair_id` joins work: 531 BIPIA and 17
InjecAgent pair_ids, 518 BIPIA pairs carrying all three tiers.

| Corpus | rows | | Corpus | rows |
|---|---|---|---|---|
| BIPIA | 1,200 | | InjecAgent | 186 |
| BIPIA-Clean | 518 | | InjecAgent-Clean | **170** |
| BIPIA-HardNeg | 531 | | InjecAgent-HardNeg | 510 |

Two defects in the output, both now fixed:

**1. InjecAgent-Clean came out 170, not ~510.** `FILLER_CLEAN` held only 10 distinct strings, so
`INJAG_CLEAN_PER_TMPL = 30` yielded 10 unique rows per template and the rest de-duplicated away.
The cap was a lie. Pool expanded to 40 declarative sentences.

**2. Length parity was broken, in the direction that matters.**

| | injected | clean | hard-neg |
|---|---|---|---|
| BIPIA (median chars) | 852 | 950 | **1,044 (+23%)** |
| InjecAgent | 385 | 316 | 329 |

v14 Exp 3 measured length as worth −0.0137 AUC — small but not zero. A tier-2 false positive at
a 23% length gap is unreadable: is the detector responding to the *instruction*, or to the
document being longer? Fillers are now assembled to match the length of the payload they stand
in for (`build_filler`: nearest-length first piece by bisect, rotating within a window for
variety, further pieces appended only while under budget). Verified against the real length
distribution — **≥99% of outputs land in [0.7, 1.4]× target at every payload decile**, where the
naive rotation was 2.4× too long at the 10th percentile and 20% short at the 90th.

v15a now prints a **GAP warning** if any tier drifts more than 10% from the injected median, so
this cannot pass silently again.

**Also noted:** 13 of 531 BIPIA pair_ids have no tier-1 twin — their clean carrier text collided
with another row after `clean_text()` and was de-duplicated. Harmless, but v15c's paired-margin
analysis must restrict to complete pair_ids.

**Extraction scope after run 1:** 3,115 indirect rows + 400 Dolly = 3,515 → **~117 min**, close
to the estimate.

## v15a run 2 (2026-09-27) — the filler was never the problem

Run 2 length-matched the fillers per row and **the BIPIA gap grew to 26.3%** (injected median
862 chars, clean 963, hard-neg 1,089). Per-row filler lengths were fine — payload median 74 vs
filler median 62, p90 280 vs 267. So the insert matched and the document still did not.

The carrier population was the culprit:

| carrier chars | pair_ids | injected rows each |
|---|---|---|
| 0–400 | 38 | **3.39** |
| 400–800 | 164 | 3.01 |
| 800–1500 | 205 | **1.71** |
| 1500–3000 | 124 | 1.82 |

`cid` was hashed from the carrier **after** truncation to `MAX_CHARS − len(payload)`. One long
source document, truncated against payloads of different lengths, split into several cids —
while a short carrier never truncates and stays one cid with ~3 injected rows. Negatives were
emitted once per cid, so long carriers were over-represented among them: median carrier 963
against 758 for injected.

### Fix, part 1 — the corpus
`cid` is now hashed from the **untruncated** carrier. One source document, one `pair_id`.

### Fix, part 2 — the metric, and this is the part that matters

The rest cannot be fixed in the corpus, and a sampling scheme that pretended otherwise would be
worse than leaving the mismatch visible. **The clean tier's text *is* the carrier**, so there
can only ever be one clean row per carrier. Its carrier distribution is uniform over distinct
carriers by construction, while the injected tier has repeats. No re-weighting of a
one-row-per-carrier tier can reproduce a distribution that contains repeats.

So the unit of analysis becomes the **carrier**:

- v15a records `carrier_n_inj` in `meta`
- v15c collapses injected rows to one mean score per `pair_id`
- the headline ASR/FPR is the **per-carrier** pair; row-level figures are still printed for
  continuity with v14 but are not the number to quote

Both tiers then sit on an identical carrier population and the paired comparison — the same
document, with and without the payload — is exact rather than approximate. It is also the
honest unit: a detector is deployed against documents, not against rows of a sampling table.

v15a's verification now prints length parity **per row** (expected to differ) and **per carrier**
(must be small), so the distinction cannot be lost again.

**Also fixed:** InjecAgent fillers now sample the payload length *distribution* rather than its
median — run 2 produced fillers at 109/111/114 chars (p10/med/p90) against real payloads
spanning 73–187, and a constant length is itself a signal a tree can learn.

## v15a run 3 (2026-09-27) — parity gate passed, two further fixes

13,245 rows. **BIPIA per-carrier length gap 26.3% → 0.5%.** The carrier-population fix works.
Spans located 100%; 480 of 481 BIPIA pair_ids carry all three tiers; all 17 InjecAgent do.

| Corpus | inj | clean | hard-neg |
|---|---|---|---|
| BIPIA | 1,200 | 480 | 481 |
| InjecAgent | 186 | 479 | 509 |

### The InjecAgent "60.5% gap" was the check, not the corpus
InjecAgent's template *requires* a filled slot, so its tier 1 **replaces** the payload rather
than removing it — unlike BIPIA, whose tier 1 is the bare carrier. No span was recorded for
those rows, so the gate fell back to full text length and compared clean's whole document
against injected's document-minus-payload. Real per-carrier totals: inj 369 / clean 406 / hard
411, on 17 templates. Noise. Fixed by recording `span_of` on the clean tier.

### Fixed stride would have gutted the version

Predicted chunks per row at the planned 32-token stride:

| corpus | median chunks | rows under 3 |
|---|---|---|
| BIPIA | 6.7 | 5.6% |
| **InjecAgent** | **3.1** | **43%** |
| InjecAgent-Clean | 3.0 | 49% |
| InjecAgent-HardNeg | 3.0 | 53% |

InjecAgent documents are ~98 tokens. Three chunks means **two deltas**, and `max − median` over
two values is degenerate. v15's core statistic would have been structurally dead on the family
with ASR 19.35% that v15 exists to fix. BIPIA was fine, which is exactly how this nearly passed
unnoticed — the aggregate row count looked healthy.

**Proportional chunking** replaces the fixed stride:
`n_chunks = clamp(n_tok // 8, 2, 16)`, boundaries spaced across each row's real tokens.

| n_tok | proportional | fixed-32 |
|---|---|---|
| 70 (InjecAgent p10) | **8** | 3 |
| 98 (InjecAgent med) | **12** | 4 |
| 215 (BIPIA med) | **16** | 7 |
| 512 (cap) | 16 × 32 tok | 16 |

Cost and storage are unchanged. Two side benefits: order statistics become directly comparable
across documents because every row contributes the same number of deltas, and `argmax/K` is a
true **relative** position rather than an absolute token offset.

### An off-by-one that would have read as "the method does not work"

`np.diff` gives MC−1 deltas for MC segments, so segment 0 had no delta — and **BIPIA places the
payload at the start half the time**. Those rows would have been mapped to delta 0, which is
segment *1*. Exp 3 would have shown roughly 50% localisation failure, indistinguishable from the
method genuinely failing.

Segment 0's delta is now its state relative to the baseline mean — the step from *no content* to
*segment 0* — so deltas align 1:1 with segments. `payload_chunk` now uses relative position ×
chunk count, which also drops the chars-per-token proxy entirely.

## A self-match leak in Exp 3, caught on synthetic data

The final pre-flight dry run fed the whole v15c pipeline **pure random arrays**. Exp 3 came back
at **100.0% exact localisation**.

Prototypes were taken in-sample from every injected row's own payload delta, so each row matched
itself and the argmax landed on the correct chunk by construction. Exp 3 is described in this
README as "the one result a better classifier cannot manufacture" — and it was manufacturing it.

| Exp 3 on pure noise | layer 4 | layer 8 |
|---|---|---|
| in-sample prototypes (old) | **100.0%** | **100.0%** |
| grouped-out + self-masked (now) | 6.7% | 8.3% |

Uniform chance on that synthetic set is 6.9%, majority baseline 10.6%. The fixed version returns
chance, as it must.

Prototypes are now grouped out by `payload_id` across 5 folds, with the self column masked as a
second line of defence.

**This is the third self-match leak in this project** — v11's positional masking, v15c's
`features()` comparing against the max instead of removing the column, and now Exp 3. The
standing rule added to AGENTS.md: assume every similarity-to-prototype computation has one, and
noise-test every headline statistic before reporting it.

## Pre-flight (all green)

| Check | Result |
|---|---|
| BIPIA carrier parity | **0.5%** |
| InjecAgent carrier parity | **0.0%** |
| Spans located | 100% |
| pair_ids with all 3 tiers | 480/481 BIPIA, 17/17 InjecAgent |
| Chunks: BIPIA | median 16, min 8 |
| Chunks: InjecAgent | median 12, min 4, **0% under 3** |
| BIPIA payload position | 596 start / 604 end |
| Exp 3 majority baseline | **49.7%** — the number to beat |
| Extraction | 3,735 rows, ~124 min, 3.9 GB |
| Disk free | 538 GB |
| Noise test | Exp 1–4 return chance |

## v15b complete (2026-09-28)

3,735 rows, 15 shards, **no gaps, no duplicates, all finite**, 3.34 GB. Manifest written.

Realized chunk counts — better than predicted, because real tokenization gives InjecAgent 150
median tokens against the 98 estimated from chars/4:

| corpus | median chunks | min | median tokens |
|---|---|---|---|
| BIPIA | 16 | 14 | 322 |
| BIPIA-Clean | 16 | 13 | 360 |
| BIPIA-HardNeg | 16 | 14 | 379 |
| InjecAgent | **16** | 10 | 150 |
| InjecAgent-Clean | 16 | 10 | 141 |
| InjecAgent-HardNeg | 16 | 11 | 142 |
| Baseline-Dolly | 6 | 5 | 55 |

**0 rows under 3 chunks.** 15.3% hit the 512-token cap.

### PCA baseline caveat
The basis and Mahalanobis covariance are fit on **2,789** Dolly chunk states — 5.4:1
samples-per-dimension, better than v14c's 3.9:1. But Dolly prompts are 55 tokens median against
BIPIA's 322, so the baseline represents *short-document* chunk positions better than late
positions in long documents. Not a blocker, and worth stating in the write-up; a few hundred long
baseline documents would close it if an examiner presses.

## v15c run 1 crashed — my bug, not the data

`KeyError: 'v14_lasttoken'` at the per-carrier summary. The block was inserted **before** the loop
that fills `E1`, because my `str.replace` matched the *first* `print("-"*77)` — the table header —
rather than the one after the loop. Nothing was wrong with the extraction or the corpus.

Fixed: premature copy removed, a dead `hardneg_by_corpus ... for c in []` line dropped, and an AST
use-before-define pass now runs clean across all nine cells.

The recurring lesson for this version: **anchor notebook patches on a unique string and assert the
match count.** Three of v15's bugs came from `.replace()` hitting the wrong copy.

## Results (2026-09-28) — the premise is falsified

3,335 evaluation rows, 2,411 payload groups, PCA 512, HistGradientBoost.

### Exp 1 — position features are worse than the whole-document state

| arm | features | row AUC | **per-carrier AUC** | per-carrier ASR % | FPR tier-1 | FPR tier-2 |
|---|---|---|---|---|---|---|
| `v14_lasttoken` (Mahalanobis only) | 8 | 0.9024 | **0.7851** | 65.83 | 10.32% | 31.10% |
| `v15_position` | 96 | 0.7198 | **0.5325** | 94.45 | 10.13% | 5.54% |
| `both` | 105 | 0.8927 | **0.7815** | 66.86 | 10.12% | 28.24% |

Ninety-six position-derived features lose to eight whole-document Mahalanobis scalars, and
adding them to the whole-document features makes things slightly *worse*. Detection under
`v15_position` alone: BIPIA 32.3%, InjecAgent **3.8%**.

**Per-carrier AUC 0.7851 vs row-level 0.9024** for the same arm: a large part of the row-level
score was carrier-level variation, not payload detection. The per-carrier unit was introduced
precisely to expose that, and it did.

### Exp 3 + the v15d/v15e diagnostics — a payload is *not* a localised disturbance

Exp 3 came back at 48.7% exact against a 43.0% majority baseline — a +5.7 lift, flat across all
eight layers. Flat across depth is itself suspicious: v14's diagnostic put layer 4 at ~3× the
margin of any other depth, so a real trajectory signal should vary.

`code/v15d_diag.py` found why. BIPIA places half its payloads at **segment 0**, and the
segment-0 delta is `Z_0 − MU` — a *magnitude*, not a *step*. Measured:

| | mean cosine across rows |
|---|---|
| `D_0` vs `D_0` | **+0.24 to +0.36** |
| `D_k` vs `D_k` (steps) | +0.04 to +0.09 |
| `D_0` vs `D_3`, same row | +0.006 |

`D_0` vectors form a tight cluster; step vectors are near-orthogonal. So the max-cosine argmax
is dragged to segment 0, and "exact" tracks the share of start-payloads rather than any
localisation. My own segment-0 fix created this.

`code/v15e_localise.py` removes the confound — **BIPIA end-payloads only**, so no `D_0` in the
ground truth or the prototype pool — and compares against a constant predictor:

| layer | localiser | exact | within 1 | vs constant |
|---|---|---|---|---|
| 4 | max-cos to prototypes | 40.1% | 76.3% | −3.6 |
| 8 | max-cos to prototypes | 43.0% | 79.5% | −0.7 |
| 16 | max-cos to prototypes | 40.6% | 81.0% | −3.1 |
| 31 | max-cos to prototypes | 30.5% | 61.9% | −13.2 |
| 4 | argmax ‖D_k‖ (untrained) | 5.6% | 12.6% | −38.1 |
| 16 | argmax ‖D_k‖ (untrained) | 8.4% | 27.0% | −35.3 |
| — | **constant = modal segment** | **43.7%** | **84.8%** | — |

**No localiser beats a constant guess**, on exact or within-1, at any depth. And `argmax ‖D_k‖`
— "the payload is the biggest state change", which needs no training at all — sits at uniform
chance (~6%). The segment where the model's state moves most is not the segment containing the
payload.

> **The finding: in this representation, an injected payload is not a localised disturbance of
> the layer trajectory.** The signal that v14 reads off the last token is a *whole-prompt*
> property, integrated across the context — not a step change at the moment the payload is read.

That is a clean falsification of v15's premise, and it is a reportable result rather than a
failed experiment.

### The second finding: it detects *instructions*, not *injections*

Tier-2 hard negatives — a benign Dolly imperative inserted at the same position, length-matched
— are the control no earlier version had:

| arm | BIPIA-Clean (tier 1) | **BIPIA-HardNeg (tier 2)** | BIPIA detection |
|---|---|---|---|
| Mahalanobis only | 19.38% | **40.75%** | 59.67% |
| both | 19.79% | **52.39%** | 71.08% |

The detector fires on a *benign imperative sentence in untrusted content* at 41–52%, against
19–20% on the untouched carrier and 60–71% on real injections. The margin between "an
instruction is present" and "this instruction is an injection" is narrow.

> **The detector is substantially detecting the presence of an instruction inside untrusted
> content, not the maliciousness of that instruction.**

This is arguably v15's most valuable output. It reframes what latent-space injection detectors
actually do, it is directly testable by others, and no number in v02–v14 could have revealed it.

### Exp 2 is mis-specified — do not cite it

| arm | spread | contexts |
|---|---|---|
| Mahalanobis only | 9.83× | 2 |
| position | 1.36× | 2 |
| both | 24.5× | 2 |

v14's 7.5e3× spanned **eight** benign corpora, from XSTest (0.000132) to InjecAgent-Clean
(0.994230). The v15 chunk cache contains only the two indirect contexts, so this measures a
different quantity. Restricted to those same two contexts, **v14's spread was already 1.24×** —
so the "collapse" is an artifact of dropping the corpora that created the spread, and the
last-token arm is in fact *worse* here than v14 was. Testing the calibration premise properly
would need the short-prompt benign corpora chunk-extracted; since position features do not work,
that is moot.

### Exp 4 — indirect LOCO

| arm | held out | AUC | ASR % |
|---|---|---|---|
| Mahalanobis only | BIPIA | 0.6365 | 74.64 |
| Mahalanobis only | InjecAgent | **0.8400** | 42.47 |
| position | BIPIA | 0.5237 | 87.11 |
| position | InjecAgent | 0.5467 | 87.59 |
| both | InjecAgent | 0.8131 | 51.42 |

Position features are at chance across families. Eight Mahalanobis scalars roughly match v14's
full 24-feature set (0.7031 / 0.8042) against a harder negative set.

## A labelling error to correct before citing anything

The arm named `v14_lasttoken` is **not v14's feature set** — it is only the 8 Mahalanobis
features (`M_L*`). v14 used those plus 16 prototype max-cosine features. So the ablation is
"96 position features vs 8 whole-document scalars", not "v15 vs v14". That makes the position
result *more* striking, not less, but the arm should be renamed `mahalanobis_only` in the
write-up.

## v16 — and it needs no new GPU time

The non-indirect rows are byte-identical between the v14 and v15 corpora (**9,910/9,910 exact
text match**), and the last-token state is the final chunk of the v15 cache. So both caches can
be joined with no re-extraction:

- v14 cache → last-token states for every direct family and all eight benign corpora
- v15 cache → last-token slice for the indirect families **and the tier-2 hard negatives**

**The v16 experiment: retrain v14's detector with the hard negatives in training.** Finding (b)
says the detector cannot separate a benign instruction from an injection — but it was never
shown one during training. If it can learn that boundary, ASR and FPR both improve and the
limit softens. If it cannot, finding (b) is a hard property of the representation and becomes a
much stronger claim.

Either outcome is publishable, it costs no extraction, and it is the natural next step.

## Status

v15's premise is falsified. The thesis's positive result remains **v14 run 2** — the classifier
finding, the solved v02 confound, and ASR 7.09% with every benign context at 10% FPR. v15 adds
two negative findings that are worth a chapter between them. Fill in after v15c, then update `EXPERIMENTS.md` and `AGENTS.md` (R2/R3/R5).

| Metric | v14 run 2 | v15 Exp 1 (both) | Target |
|---|---|---|---|
| ASR % [Wilson] | 7.09 (context-matched) | | < 5% |
| Worst tier-1 FPR % | 47.33 (BIPIA-Clean) | | < 10% |
| Tier-2 FPR % | — (did not exist) | | report only |
| Threshold spread | 7.5e3× | | ~1× |
| Localisation vs majority | — | | > 0 |
| Indirect LOCO AUC | 0.703 / 0.804 | | — |
