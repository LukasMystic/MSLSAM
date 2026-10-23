# 16 — MSLSAM v17 (Baseline Comparison)

**Status:** ✅ run 2 COMPLETE (all 12 methods) — both pre-registered verdicts FIRED against MSLSAM; payload-leak diagnosis below · **Source:** `v17a_baselines.ipynb` → `v17a2_judge.ipynb` (**fresh kernel**) → `v17b_compare.ipynb`

## Why v17

MSLSAM reports SPML ASR 0.91%. **Nobody knows whether that is good**, because no other detector
has ever been run on this corpus. A regex for *"ignore previous instructions"* might take 40% of
SPML for free, or 5%. Until that floor exists, the headline floats — and this is the first thing
an examiner will ask.

## The four comparators

| tier | method | params | gated |
|---|---|---|---|
| floor | keyword / regex rules | 0 | — |
| small | `protectai/deberta-v3-base-prompt-injection-v2` | 184M | no (Apache-2.0, **archived upstream**) |
| small | `meta-llama/Llama-Prompt-Guard-2-86M` | 86M | **yes** — skipped cleanly if no access |
| ceiling | **Llama-3-8B judging itself** | 8B | — |

### The judge is one forward pass, not a generation

Asking the model *"is this a prompt injection?"* and reading the **logit gap between `Yes` and
`No` at the first generated position** gives a continuous score from a single forward pass — no
sampling, no decode loop.

That makes the central comparison exact rather than rhetorical: **MSLSAM and the judge cost the
same one forward pass of the same model.** The question becomes which reads the prompt better —
the hidden states, or the model's own answer about them.

### The rules are deliberately not tuned

18 patterns from published guardrail heuristics — override phrasings, role switches,
instruction-disclosure requests, exfiltration verbs, and German equivalents for DeepSet's 348
German rows. **Not tuned on this corpus, and must not be: a floor you tuned is not a floor.**
Scored as a count of distinct patterns matched so it can be thresholded like everything else,
and *also* reported as a binary filter, which is how a rule engine is actually deployed.

## The asymmetry, handled rather than hidden

MSLSAM is **trained on this corpus**; every baseline is **zero-shot**. Comparing them directly
flatters MSLSAM. So v17b reports two MSLSAM rows:

| row | meaning | fair against zero-shot? |
|---|---|---|
| **MSLSAM (LOCO)** | that family removed from training entirely | **yes** — neither has seen it |
| MSLSAM (in-dist) | grouped CV, family seen | no — upper bound only |

**The LOCO row is the one that belongs in the thesis next to the baselines.** From v16 we
already know LOCO is brutal: SPML goes 0.91% → 62.81% when its own family is held out. If a
184M classifier beats that zero-shot, the honest contribution is the *cost* argument and the
in-distribution ceiling, not raw detection — and v17b prints exactly that conclusion when it
happens.

Every method is scored through the **same nested per-context threshold protocol** at a 10% FPR
budget, so the ASR column is directly comparable.

## The four questions

1. **Does MSLSAM beat the floor?** If regex matches it on SPML, the headline collapses. v17b
   prints a `WARNING` line if rules match or beat MSLSAM on any family.
2. **Does it beat a dedicated classifier, zero-shot vs zero-shot?**
3. **Does it beat the same model judging itself on the same one forward pass?**
4. **Do the baselines share the instruction/injection confusion?** v16 found MSLSAM fires on
   tier-2 hard negatives at 57–79%. If ProtectAI and the judge do too, that confusion is a
   property of **the task**, not of MSLSAM — a stronger and more publishable finding than a flaw
   in one method. This has, as far as I know, not been measured before, because nobody else has
   a length-matched benign-imperative control.

## Two cost columns, and the difference matters

| column | question it answers |
|---|---|
| `total_ms` | compute spent on detection alone — **a small classifier wins this outright** |
| `marginal_ms` | compute *added* when the protected 8B model is already processing the prompt |

MSLSAM reuses the protected model's forward pass, so its marginal cost is **3.22 ms (+0.16%)**.
Every other method is a separate model and adds its full cost. The judge is the sharp case: same
model, same single pass, but a **second** pass with its own wrapper prompt — ~2,000 ms marginal
against MSLSAM's 3.22.

**Both columns are printed and both belong in the write-up.** Quoting only the marginal one
would be an overclaim, and my earlier framing of "+0.16%" did exactly that without the
qualifier.

## Cost to run

- **v17a**: rules are instant; the two small classifiers are minutes; the judge is one forward
  pass per row (~0.5 rows/s), defaulting to a **stratified 3,000-row subsample ≈ 100 min**. Set
  `JUDGE_N = None` for the full 11,245 rows (~7 h).
- **v17b**: cache-only and fast, but it refits MSLSAM five times (one grouped CV plus four LOCO
  folds) from the `joined_v16` cache.

## If Prompt Guard is gated

`meta-llama/Llama-Prompt-Guard-2-86M` requires access approval. v17a catches the failure, prints
what to do, and continues; v17b drops any missing method from the table without complaint.

## Pre-flight

Both notebooks compile; the AST use-before-define pass is clean over all 12 code cells; the
DataFrame-column-attribute grep (the check that would have caught v16's crash) is clean; and a
synthetic dry-run confirms the NaN-subsample path (the judge scores a subset), that hard-negative
cross-scores populate only for families that have them, and that a missing baseline column does
not break the comparison table.

## v17a run 1 — aborted at 0.07 rows/s, two bugs fixed

The judge loop ran at **0.07 rows/s** (681 min ETA) against v15b's 0.5 rows/s on the same model.
Both causes were mine, and both were problems this project had already solved:

**1. The full `CausalLM` call.** `jmdl(**enc).logits` materialises logits for *every* position
across the full 128k vocabulary — for B=4 at 512 tokens that is a **525 MB** fp16 tensor per
forward, from which we read two numbers. v15b deliberately called `base_m(**enc)` to skip the
`lm_head` for exactly this reason. The judge now runs the base transformer, takes the last
position, and projects it onto just the two token rows of `lm_head`: **(B, 2) instead of
(B, 512, 128256)** — 16 bytes plus a 33 KB hidden state.

**2. Fixed `BS=4` in corpus order** — the v14b padding-waste bug, reintroduced. Measured on the
real length distribution: **padding waste 1.582× → 1.006×**, batches 751 → 680.

The 681-minute ETA was also computed from the first batch, which on MPS includes kernel
compilation, so 0.07 was never the steady-state rate. The loop now reports a **rolling
per-shard** rate instead.

> **Lesson for the write-up and for any future version:** when a new notebook runs the same
> model as an old one, copy the old one's forward path verbatim. Both bugs were solved problems
> that I re-introduced by writing fresh code.

### Two further improvements made at the same time

**Shard checkpointing.** The judge writes `results/v17/judge_shards/jsh_*.npz` and resumes from
them, so an interrupted or slow run loses nothing.

**A floor of 150 rows per corpus** in the judge subsample, replacing pure proportional
allocation. Proportional would have given InjecAgent 50 rows and JBB-Benign 27 — the same
few-dozen-row estimate that made v14's "InjecAgent FPR 19.35%" meaningless (it came from 17
negatives). With the floor, `JUDGE_N = 3000` resolves to 3,487 rows and every corpus gets at
least 150, or all of it:

| corpus | judged / total | | corpus | judged / total |
|---|---|---|---|---|
| SPML | 667 / 2,499 | | InjecAgent | 150 / 186 |
| WJB-Adv-Benign | 667 / 2,500 | | InjecAgent-Clean | 150 / 479 |
| BIPIA | 320 / 1,200 | | InjecAgent-HardNeg | 150 / 509 |
| SPML-Benign | 320 / 1,198 | | DeepSet | 150 / 263 |
| WJB-Vanilla-Benign | 213 / 800 | | JBB-Benign | 100 / 100 (all) |

## v17a run 2 — stalled 48 minutes, zero shards; pipeline split in three

The judge sat at `shards 7 x 500 | already done 0` for **48 minutes** (shard folder created
08:36, nothing written by 09:24). With 500-row shards and no intra-shard reporting, that could
not be told apart from a hang. Causes, all mine:

1. **Progress only printed when a 500-row shard finished** — the silence was the design.
2. **`use_cache=False` dropped** — v15b passes it; the judge built and discarded a 32-layer KV
   cache every batch. The *second* time in one version a solved v15b problem came back because
   the forward path was rewritten rather than copied.
3. **The judge shared a kernel with ProtectAI and Prompt Guard.** On a 24 GB machine where Llama
   fp16 is ~16 GB, leftover allocations plus an open browser push it into swap.

| notebook | kernel | runtime | writes |
|---|---|---|---|
| `v17a_baselines` | any | minutes | `scores/rule_*.npy`, `protectai_deberta.npy`, `promptguard2_86m.npy` |
| `v17a2_judge` | **fresh, alone** | ~2 h | `scores/llm_judge.npy`, `_latency_8b.json`, `judge_shards/` |
| `v17b_compare` | any | minutes | `metrics.json`, cost-vs-ASR figure |

Each method writes the moment it finishes; re-running v17a skips anything already scored. The
judge copies v15b's forward path verbatim, prints progress every 20 batches, uses 250-row
resumable shards, runs an untimed MPS warm-up, and runs a **32-row smoke test that refuses to
start below 0.2 rows/s**.

## Cost is now LATENCY, not borrowed throughput

MSLSAM's forward pass was previously taken as **2,000 ms — v15b's *batched* throughput**, never a
latency. Every small classifier is now timed at **batch size 1** (median of 50, after warm-up),
and v17a2 times a batch-size-1 forward in **MSLSAM's exact format** and in the judge's format on
the same loaded model. v17b warns loudly if the 8B measurement is missing. The methods document's
cost section (§8) uses the old 2,000 ms figure and must be updated after v17 runs.

## ProtectAI is archived — kept, deliberately

Still downloadable and still the most-used open detector (~813k downloads last month); a frozen
baseline is *more* reproducible; its published PINT score (79.14%) anchors this corpus
externally. Commit hash and access date are recorded in `scores/protectai_deberta.cost.json` and
carried into `metrics.json` under `provenance`. **Prompt Guard 2 (86M) is the maintained
counterpart.**

## Pre-flight

AST clean across all three notebooks; column grep clean; dry-run of the full handoff including a
**simulated interruption after 3 shards and a resume** that completed only the remaining shard
with every row aligned to its own text; and v17b's per-method loader, including the
`rule_count → rule_keyword` cost lookup — a real bug in the previous v17b that would have left
the rules' cost blank.

## Results — run 1 (2026-09-28), corrected for a tie-handling bug

### The bug: v17b's first verdict was FALSE

v17b run 1 printed *"MSLSAM does NOT beat keyword rules zero-shot"* plus four `WARNING` lines
saying rules matched or beat MSLSAM on every family. **None of that was real.** The keyword-rule
score is a count; **100% of SPML-Benign scores exactly 0**, so the 90th-percentile threshold is 0,
and with a `>=` rule every prompt is flagged — **ASR 0%, FPR 100%**. The verdict read only the ASR
column. Fixed with a **tie-aware threshold** (step up to the next distinct value, which is within
budget by construction) and a verdict that only considers methods whose **worst own-context FPR
is ~10%**. Re-scored: only the rules changed; every neural method moved < 1.5 ASR points. The
buggy run is archived as `results/metrics_run1_TIEBUG.json`. v14–v16 used the same threshold
function, but their scores are near-continuous and their reported FPRs were ~10%, so unaffected.

### Corrected: ASR % at each family's own 10%-FPR threshold

| method | trained here? | SPML | DeepSet | BIPIA | InjecAgent | **pooled** |
|---|---|---|---|---|---|---|
| keyword rules | no | 87.15 | 84.79 | 98.75 | 79.03 | **90.00** |
| Prompt Guard 2 (86M) | no | **1.59** | 20.23 | 89.43 | 52.21 | **30.45** |
| Llama-3-8B judge ᵃ | no | 5.80 | 14.73 | 88.78 | **8.87** | **27.83** |
| **MSLSAM (LOCO)** | **no — family held out** | 49.97 | 35.59 | 81.73 | 57.16 | **58.57** |
| MSLSAM (in-dist) | yes | **0.96** | **8.54** | **36.35** | 65.89 | **14.59** |

ᵃ judge scored on a stratified 3,487-row subsample (≥150 per corpus), so its CIs are wider.
ProtectAI DeBERTa: **still missing** (tokenizer failure; v17a not yet re-run after the fix).

### What it says — and the pre-registered verdict fired

1. **Zero-shot vs zero-shot, MSLSAM loses.** Prompt Guard 2 (30.45%) and the Llama judge
   (27.83%) both beat MSLSAM-LOCO (58.57%). The README stated before the run what this means:
   *the contribution becomes the cost argument plus the in-distribution ceiling, not raw
   detection.*
2. **The SPML headline is not unique.** MSLSAM trained on SPML reaches 0.96%; **Prompt Guard 2,
   never trained on this corpus, reaches 1.59%.** "Target met on direct injection" is true but
   an off-the-shelf 86M classifier nearly meets it too.
3. **MSLSAM in-distribution is best on three families** — and dramatically on BIPIA (36% vs ~89%
   for everything else). **But it is the only method trained on this data**, so this cannot yet
   be credited to hidden states rather than to in-domain training. It needs a *trained* text
   baseline on the same grouped folds.
4. **InjecAgent is the Llama judge's**: 8.87% ASR — but see below for what that costs.

### Hard negatives at per-context thresholds — finding (b) needs restating

FPR on tier-2 hard negatives (benign imperative, same position, length-matched):

| method | BIPIA-HardNeg | InjecAgent-HardNeg |
|---|---|---|
| keyword rules | 2.1% | 0.2% |
| Prompt Guard 2 | 9.7% | 24.9% |
| Llama-3-8B judge | 8.8% | **66.5%** |
| MSLSAM (LOCO) | 7.8% | 19.8% |
| MSLSAM (in-dist) | 16.9% | **7.1%** |

v15/v16 reported MSLSAM flagging hard negatives at **40.75–79.26%**. Those were measured at a
**pooled** threshold set mostly by short benign text, which sits far below retrieved documents.
At a **per-context** threshold MSLSAM flags them at **7–17%**. So the dramatic version of finding
(b) was largely threshold placement. The instruction-vs-injection confusion is real but moderate
for MSLSAM — and **strongest in the Llama judge on InjecAgent (66.5%)**, which is how it earns its
8.87% ASR there. It is a property of the task that every method trades against differently.

### Cost — batch-size-1 latency, warm, measured on this machine

| method | own pass | marginal on an 8B pass already running |
|---|---|---|
| keyword rules | 0.09 ms | 0.09 ms |
| Prompt Guard 2 | 241.6 ms ᵇ | 241.6 ms |
| Llama-3-8B judge | 378.5 ms | 378.5 ms (a second pass) |
| **MSLSAM** | 326.0 ms forward + 3.22 ms | **3.22 ms (0.99%)** |

ᵇ timed **cold** under the old method; re-measured warm on the next v17a run (`REDO`). Prompt
Guard 2 reports **279M parameters**: "86M" is its backbone; the rest is a 250k-token multilingual
embedding table. Say so in the write-up.

**Correction to the methods document:** it used ~2,000 ms for the 8B forward pass and "+0.16%".
That was v15b's *batched throughput*. Measured batch-size-1 latency is **326.0 ms**, so MSLSAM's
marginal overhead is **0.99%**, not 0.16%. Still small; six times larger than stated.

### The judge run itself

Probe cold **0.66** / warm **0.67** rows/s, MPS driver 16.10 GB → 16.48 GB (clean). 3,487 rows in
~37 min. **Cold ≈ warm means shape recompilation was NOT the dominant cost** — my diagnosis of
run 3's 0.13 rows/s is not supported by this evidence. The likelier cause was memory pressure;
this run started clean. The padding bucketing stays (harmless, fewer shapes), but it should not be
credited with the speed-up.

### Still to do
- Re-run **v17a** (ProtectAI after `pip install sentencepiece protobuf`; Prompt Guard 2 warm latency),
  then **v17b**.
- **A trained text baseline on the same grouped folds** — the only way to tell whether MSLSAM's
  in-distribution lead is the method or the training data.
 Fill in after v17b, then update `EXPERIMENTS.md` and `AGENTS.md` (R2/R3/R5).

| method | SPML | DeepSet | BIPIA | InjecAgent | pooled ASR | total ms | marginal ms |
|---|---|---|---|---|---|---|---|
| keyword rules | | | | | | | |
| ProtectAI DeBERTa | | | | | | | |
| Prompt Guard 2 86M | | | | | | | |
| Llama-3-8B judge | | | | | | | |
| **MSLSAM (LOCO)** | | | | | | | |
| MSLSAM (in-dist) | | | | | 12.21 (v16) | 2003 | 3.22 |


## v17c — trained baselines on MSLSAM's exact folds (built 2026-09-28, run 2026-09-28)

Run 1 left one confound standing: MSLSAM in-dist led on SPML, DeepSet and BIPIA, but it was the
**only method trained on this corpus**. v17c trains text baselines on the **identical folds**:

| baseline | question |
|---|---|
| **TF-IDF + LR** (word 1–2g + char_wb 3–5g, C=4, untuned) | is the lead just in-domain training? |
| **MiniLM probe** (frozen `all-MiniLM-L6-v2`, 22M, mean-pooled) + LR **and + HGB** | is it Llama's hidden states, or would any learned encoder do? |

The MiniLM probe is MSLSAM's structural twin (frozen representation, trained head) and gets
**MSLSAM's own HGB head** too — v14 showed the head was the bottleneck, so denying the baseline
that head would rig the comparison. `max_length=512` so BIPIA's end-of-document payloads are not
truncated away.

**Identical folds, verified:** on the real 11,245 rows, v17c's split is bit-identical to v17b's
MSLSAM split (fold sizes 2247/2250/2251/2250/2247; zero carrier groups straddle folds). v17c saves
`scores/_folds_indist.npy` and **v17b asserts its own MSLSAM folds match before comparing.**
Vectoriser, scaler and head are fitted on the training fold only.

**v17b now runs Exp 1b — trained vs trained, paired:** per family, a McNemar exact test on the
SAME injections, each method at its own 10%-FPR threshold, so FPR is held fixed and only misses
are compared. Pooled ASR is descriptive; per-family claims rest on McNemar.

**Pre-registered reading:** TF-IDF in-dist within **2 pooled-ASR points** of MSLSAM in-dist
(14.59%) or better → MSLSAM's in-distribution lead is the **training data**, not the hidden
states. Ahead by more → a *candidate* representational advantage, claimable only for families
where McNemar says MSLSAM at p < 0.05. MiniLM ≈ MSLSAM → the advantage belongs to "a learned
representation", not to Llama. **BIPIA is the family to watch.**

**Install cell:** v17a now starts with a cell that `%pip install`s `sentencepiece protobuf` into the
kernel's own environment if missing, and says whether a restart is needed (only if `transformers`
was already imported in that kernel).

**Run order:** v17a (install cell → rest; re-does Prompt Guard 2 warm latency, retries ProtectAI)
→ v17c → v17b. The judge (v17a2) does not need re-running.

### v17c run 1 — crashed after the TF-IDF fits (my bug, fixed)
`AssertionError: TF-IDF + LR: gaps in scores`. `run_protocol` required the whole LOCO array to be
filled, but WJB-Adv-Benign, WJB-Vanilla-Benign, XSTest-Safe and JBB-Benign belong to no LOCO family
and are never held out, so they are NaN **by design**, exactly as in MSLSAM's LOCO array, which
v17b's `nested()` already drops. The assertion now checks in-dist complete, LOCO complete on the
four families, and LOCO NaN elsewhere. The dry run missed it because its synthetic corpus held only
family corpora. Re-dry-run with all 15 real corpus names: passes. Nothing was saved by run 1.

## Results — run 2 (2026-09-28): all 12 methods, identical folds

ProtectAI loaded (tokenizer fallback + sentencepiece fixed it; commit 90c9989b, 184M, id2label
safe/injection, probe gap 1.000). Prompt Guard 2 warm latency re-measured. v17c ran clean after the
LOCO-assertion fix. v17b's fold check passed: trained baselines used IDENTICAL folds to MSLSAM.
Every method's worst own-context FPR is 10.5–10.7% (rules 1.9%), so every ASR below is valid.

### ASR % at each family's own 10%-FPR threshold (nested, 400 resamples)

| method | regime | SPML | DeepSet | BIPIA | InjecAgent | pooled |
|---|---|---|---|---|---|---|
| keyword rules | zero-shot | 87.15 | 84.79 | 98.75 | 79.03 | 90.00 |
| ProtectAI DeBERTa v2 | zero-shot | **1.00** | 27.47 | 84.14 | 77.44 | 30.16 |
| Prompt Guard 2 86M | zero-shot | 1.59 | 20.23 | 89.43 | 52.21 | 30.45 |
| Llama-3-8B judge (3,487-row subsample) | zero-shot | 5.80 | 14.73 | 88.78 | 8.87 | **27.83** |
| TF-IDF + LR | LOCO | 86.64 | 55.33 | 80.17 | 74.93 | 82.26 |
| MiniLM + LR | LOCO | 77.74 | 71.02 | 79.07 | 75.56 | 77.60 |
| MiniLM + HGB | LOCO | 75.85 | 73.30 | 85.55 | 78.03 | 78.59 |
| **MSLSAM** | LOCO | **50.58** | **38.72** | 80.44 | **66.55** | **59.18** |
| TF-IDF + LR | in-dist | 1.44 | 21.86 | **13.21** | **0.00** | **6.07** |
| MiniLM + LR | in-dist | 5.12 | 49.34 | 67.36 | 39.31 | 27.46 |
| MiniLM + HGB | in-dist | 1.19 | 43.00 | 47.82 | 52.28 | 19.62 |
| **MSLSAM** | in-dist | 1.13 | **8.65** | 35.35 | 66.04 | 14.42 |

Paired McNemar on identical injections, each at its own threshold:

| regime | family | only MSLSAM catches | only baseline catches | p | winner |
|---|---|---|---|---|---|
| in-dist vs TF-IDF | SPML | 31 | 26 | 0.597 | neither |
| in-dist vs TF-IDF | DeepSet | 50 | 15 | 1.6e-5 | **MSLSAM** |
| in-dist vs TF-IDF | BIPIA | 57 | 320 | 1.6e-45 | TF-IDF |
| in-dist vs TF-IDF | InjecAgent | 0 | 125 | 4.7e-38 | TF-IDF |
| LOCO vs MiniLM+LR | SPML | 859 | 182 | 1.2e-105 | **MSLSAM** |
| LOCO vs MiniLM+LR | DeepSet | 110 | 22 | 2.9e-15 | **MSLSAM** |
| LOCO vs MiniLM+LR | BIPIA | 180 | 196 | 0.439 | neither |
| LOCO vs MiniLM+LR | InjecAgent | 41 | 26 | 0.086 | neither |

### Both pre-registered verdicts fired, and both are reported as fired

1. **Zero-shot:** MSLSAM LOCO (59.18%) does not beat the best zero-shot baseline (8B judge, 27.83%).
   ProtectAI and Prompt Guard 2, which are about 10× cheaper than the Llama forward pass, reach ~30%.
2. **Trained, in-dist:** TF-IDF + LR is 6.07% pooled against MSLSAM's 14.42%, beyond the 2-point
   margin. **Pre-registered reading: MSLSAM's in-distribution lead is the TRAINING DATA, not the
   hidden states.** Also: SPML's 1.13% is matched by TF-IDF (1.44%, McNemar p = 0.60) and beaten by
   ProtectAI **zero-shot** (1.00%). The SPML headline shows that SPML is easy. It is not evidence for MSLSAM.

### Post-hoc diagnosis (NOT pre-registered): the in-dist split leaks payloads on indirect families

Checked after the run, on the real fold assignment: the share of test injections whose exact
`payload_id` also sits in the training folds.

| family | injections | unique payloads | test payload seen in train |
|---|---|---|---|
| SPML | 2,499 | 814 | **0.0%** |
| DeepSet | 263 | 172 | **0.0%** |
| BIPIA | 1,200 | 400 (250 unique strings) | **96.5%** |
| InjecAgent | 186 | 62 | **98.4%** |

`group_carrier` groups indirect rows by **carrier** (pair_id), so the same attacker instruction
appears in training inside other carriers. For BIPIA and InjecAgent, "in-dist" therefore means
"unseen carrier, **seen payload**", and a bag-of-n-grams model simply memorises the 62 or 250 payload
strings. TF-IDF wins exactly where payloads repeat (BIPIA, InjecAgent) and ties or loses exactly
where they do not (SPML, DeepSet). This explains the verdict. **It does not overturn it.**
It was found after the fact, and even restricted to the two leak-free direct families, MSLSAM
leads by only ~1.5 pooled points (1.85% vs 3.38%), which is inside the pre-registered 2-point margin.

### What survives

- **Cross-family transfer among corpus-trained methods.** Trained on the same rows, Llama hidden
  states generalise to an unseen family far better than TF-IDF or MiniLM. LOCO pooled 59.18% vs
  77.6–82.3%, significant on SPML (859 vs 182) and DeepSet. This is the one clean representational
  result. It is bounded above by the guard models, which were trained on large external data.
- **DeepSet in-dist** (unseen payloads, German/multilingual): 8.65% vs TF-IDF 21.86%, p = 1.6e-5.
  n = 263, so it is one family and a small sample.
- **Cost:** 3.22 ms marginal (0.99% of the 326 ms forward pass), against a separate 31.7 ms pass
  (ProtectAI) or 32.4 ms (PG2). Total cost still favours the small guards (~10×).
- **Hard negatives are mixed.** InjecAgent-HardNeg: MSLSAM in-dist 9.5% vs TF-IDF 23.6%.
  BIPIA-HardNeg: MSLSAM in-dist 15.9%, the worst of all 12 methods. ProtectAI is best overall on
  hard negatives (6.4% / 7.6%).

### Other run-2 notes
- **MiniLM+LR `RuntimeWarning: divide by zero / overflow in matmul`:** likely the known spurious
  warning from NumPy's Apple-Accelerate matmul. Scores checked: 0 NaN (in-dist), 0 inf, all in (0, 1),
  AUC 0.94. Not a result problem.
- **MSLSAM drifted 0.2–0.6 pts from run 1** (in-dist 14.59 → 14.42, LOCO 58.57 → 59.18). The cause:
  `PCA(512)` was unseeded, and sklearn's `auto` solver picks randomized SVD at this shape. It is now
  seeded (`random_state=SEED`). **CORRECTED by v18:** with the seed fixed at 42, MSLSAM v17 gives 15.73% (in-dist) and 52.02% (LOCO). So the seed swings it ~1.3 points in-dist and ~7 points in LOCO, not 0.2–0.6. v17's verdicts still stand, because their gaps are 8+ points.
- Archived: `results/metrics_run2.json`, `figures/mslsamv17_cost_vs_asr_run2.png`, `code/` (PCA-seeded v17b).
