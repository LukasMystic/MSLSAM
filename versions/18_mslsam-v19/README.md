# 18 — MSLSAM v19 (the fair comparison: external-only training + task-drift Δ)

**Status:** ✅ v19 COMPLETE 2026-09-29 — F1 MATCHES (4/5), **F1-LC (governing) LOSES by rule (mixed: BIPIA much better, InjecAgent much worse)**, F2 and F3 SEED-DEPENDENT (2/5), target not met **The pre-registration below is fixed now, before any v19 number exists.**

## Why

Every MSLSAM result so far had a structural advantage or disadvantage against the zero-shot guards:

- **In-family (v14–v18):** MSLSAM had trained on the family it was tested on. The guards had not.
- **Leave-one-family-out (v17, v18):** MSLSAM trained on only 3 of our 4 families. The guards were trained on
  large external datasets.

v18/v18b established that Llama's hidden states beat a text classifier when both train on the same data
(H2, stable in 5 of 5 seeds). It left the question that matters for a defence open: **trained on data from none
of the four test families, does MSLSAM match or beat the guards?** v19 asks exactly that, and adds the
**task-drift Δ** that targets indirect injection (BIPIA ~50% for every method in v18).

## Design

### Training data: external only (v19a)
6,000 rows, balanced. Nothing comes from SPML, DeepSet, BIPIA, InjecAgent, WildJailbreak, XSTest, JailbreakBench or Dolly.

| block | injections | benign | construction |
|---|---|---|---|
| direct | 1,100 hackaprompt (successful) + 400 Gandalf | 1,161 Alpaca (length-matched) + 339 NotInject (benign prompts with trigger words) | the prompt is the whole user turn |
| indirect — email | 750 LLMail-Inject attacker emails in a 3-email Enron inbox | 750 same inbox, attacker email replaced by a length-matched benign email | task = one of the LLMail challenge's user requests |
| indirect — document | 750 SQuAD paragraphs with a hackaprompt/Gandalf instruction at start/middle/end | 750 same paragraph, same position, a length-matched *declarative* sentence | task = the SQuAD question |

- hackaprompt/Gandalf payloads are split **disjointly** between the direct and document blocks.
- **Contamination guard:** an external text is dropped if ≥ 50% of its word 8-grams occur anywhere in the test set,
  or, if it is under 8 words, on an exact match. Drop counts are reported.
- **No hard-negative training** (the v16 lesson). The benign document inserts are declarative, not instructions.

### Test data: the same 11,245 rows, with indirect rows split into task + data (v19a)
- **InjecAgent:** a pure split of the existing text. The task is the user instruction; the data is `Tool result (...): ...`.
- **BIPIA:** rebuilt from source with the same carrier, the same payload or filler, and the same start/end position.
  The question becomes the task, placed before the document. v19a found that the v15 layout had cut the
  question out of 60 of 900 email/table rows. The rebuild fixes that.
- **Direct rows** (SPML, DeepSet, WJB, XSTest, JBB): unchanged, with no task.
- Every row is checked with the real tokenizer against `MAX_LEN = 1024`; **no row may be truncated**.

### Features (v19b, one forward pass per row)
User turn = `task + "\n\n" + data`, with v14b's system prompt and chat template unchanged. Hidden states at 8 layers are taken at two points:
- **anchor**: the last token of the task. For direct rows (no task), the last token of the system message, which
  is the same for every row.
- **end**: the last token of the prompt, i.e. the same position v14–v18 used.

**Δ = h_end − h_anchor** (TaskTracker's activation delta, per layer). For direct rows the anchor is constant, so
Δ is the end state shifted by a constant, and MSLSAM-Δ reduces to MSLSAM-P there.
Direct *test* rows reuse the cached v14 end states. That is allowed only if a re-extracted sample of them
matches the cache, and v19b asserts it.

### Methods (v19c). Every trained method sees the same 6,000 external rows and nothing else.
| method | input | role |
|---|---|---|
| **MSLSAM-Δ** | Δ, 8 layers, PCA(256) fitted on external train, StandardScaler, LR (balanced), C from {1e-5, 1e-4, 1e-3, 1e-2, 1e-1} chosen by inner grouped 3-fold CV | **primary** |
| MSLSAM-P-ext | h_end only, same pipeline | ablation: what Δ adds |
| TF-IDF + LR | user-turn text, v17c config | text baseline, fair training |
| MiniLM + LR | user-turn text, v17c config | encoder baseline, fair training |
| ProtectAI v2, Prompt Guard 2 | the **untrusted data** for indirect rows (their intended use); the prompt for direct rows | zero-shot guards; re-scored on the rebuilt indirect rows |
| Llama-3-8B judge | same inputs, v17's subsample | zero-shot; re-scored on the indirect rows of its subsample |
| length only | number of characters | sanity control for a length shortcut |

C's grid extends one decade below v18's, because v18 hit the grid edge in 18 of 19 splits.
That was decided from v18 and **before** v19.

Threshold protocol: unchanged from v17b. Per-family nested thresholds at a 10% FPR budget on the family's own
benign rows, 400 resamples, tie-aware. The same calibration applies to every method, guards included.

## Pre-registered reading (fixed 2026-09-29, before any v19 number exists)

Pooled ASR is the injection-weighted ASR over SPML, DeepSet, BIPIA and InjecAgent, each at its own 10% FPR.
McNemar is paired on identical injections, with p < 0.05. MSLSAM runs with PCA seeds 0–4. **Every MSLSAM
verdict below must hold in ≥ 4 of the 5 seeds**; 2–3 is reported as *seed-dependent*.

- **F1 — the fair headline.** Compare against the best full-coverage zero-shot guard, meaning whichever of ProtectAI and PG2 has the
  lower pooled ASR on the v19 test set.
  - **BEATS** if MSLSAM-Δ is ≥ 2 pooled points lower, with McNemar wins on ≥ 2 families and losses on none.
  - **MATCHES** if the pooled difference is under 2 points.
  - **LOSES** otherwise.
  - The 8B judge is compared on its subsample only (McNemar on the intersection).
- **F2 — does the task-drift Δ help indirect injection?** On BIPIA + InjecAgent pooled, MSLSAM-Δ must be ≥ 5 points below
  MSLSAM-P-ext, and McNemar must favour P-ext on neither family.
- **F3 — hidden states vs text, under fair training.** MSLSAM-Δ must be ≥ 2 pooled points below **both** TF-IDF and MiniLM,
  with McNemar wins on ≥ 2 families and no losses against each.
- **Target (ASR < 5% at the 10% budget):** reported per family. It is claimed only if the **worst** seed is < 5%.
- **Descriptive only:** hard-negative FPR (BIPIA-HardNeg, InjecAgent-HardNeg); guards scored on task+data as well
  as data only; the length-only control. If length alone reaches AUC > 0.75 on a family, that family's result is flagged.

**Cannot be changed after the run:** the external data, the C grid, the PCA dimension, the layers, the seeds, the
families, the thresholds, or the choice of the "best guard". Any change is a new version with its own pre-registration.

## Honest limits, stated in advance
- Threshold calibration uses each family's own benign rows. This is a small in-domain step, shared by every method.
- The guards' training data is only partly disclosed. ProtectAI v2 lists xstest-v2, among others. A guard may therefore have seen
  something close to our test rows. MSLSAM, by construction, has not.
- The external data covers email and documents. Tool outputs (InjecAgent's format) and tables/code (part of BIPIA)
  have **no** external training analogue. Those families are a genuine transfer test.

## Run order
1. **v19a_build.ipynb** (CPU, 5–15 min, downloads ~450 MB once). Send Claude its printed output.
2. v19b_extract (GPU, estimated 4–5 h for about 9,400 rows; resumable). Built after step 1.
3. v19b2: re-scoring of the guards and the judge on the rebuilt indirect rows (~30–40 min).
4. v19c: the analysis and the verdict (CPU).

## v19a dry run (2026-09-29)
The dry run used the real BIPIA and InjecAgent sources and the real v15 corpus. The external datasets were synthetic stand-ins
with the documented schemas, because the cloud workspace cannot reach Hugging Face. The tokenizer was a stand-in too.
- All 3,335 indirect test rows were rebuilt, and every BIPIA `pair_id` was recovered from source. Every payload and filler was found inside the rebuilt data.
- **Finding:** in the v15 corpus, **60 of 900 BIPIA email/table rows had lost their question**. `MAX_CHARS` truncation cut
  the end of `context + question`. The rebuild restores the question as the task.
- A BIPIA code traceback could exceed the token budget on its own, so the code task keeps the last 560 characters of the traceback,
  which is the actual error line.
- Direct benign rows are length-matched to the attacks. A residual gap remains because NotInject prompts are short.
  The real run prints the gap, and v19c's length-only control measures whether it matters.

### v19a run 1 — stopped at cell 4 (2026-09-29): not enough Gandalf prompts
`AssertionError: gandalf (direct half): need 400, have 342`. Lakera's Gandalf set has 1,000 rows, and after deduplication,
the length filter and the contamination guard its direct half holds 342. The fix: Gandalf supplies what it has
(~342) and **hackaprompt fills the difference**, so direct attacks stay at 1,500. The injection half is unaffected
(Gandalf + hackaprompt interleaved, 750 needed). The SQuAD rows' `source` label now records the true payload origin;
before, it was assigned by row parity. No v19 number existed at this point, so this is a data-supply adjustment, not a
change after results. The downloads succeeded, and all six sources are cached in `datasets/external/`.


## v19a — run 2 complete (2026-09-29)

| source | usable | dropped as test near-duplicates |
|---|---|---|
| hackaprompt (successful) | 17,527 of 77,936 | 102 |
| Gandalf | 684 of 1,000 | **312 (31%)** |
| LLMail-Inject Phase 2 (≤ 700 chars) | 21,449 of 90,916 | 6 |
| Alpaca | 51,974 | 8 |
| NotInject | **255 of 339** (84 near-identical after case-folding) | 4 |
| Enron ham | 14,382 | 0 |

- **External training set: 5,916 rows** (3,000 injections / 2,916 benign). Direct attacks are 1,158 hackaprompt + 342 Gandalf;
  direct benign are 1,161 Alpaca + 255 NotInject (1,416, not 1,500, because NotInject had fewer unique prompts). Email: 750 + 750 pairs.
  Document: 750 + 750 pairs. `class_weight="balanced"` absorbs the 84-row imbalance.
- **Length parity:** email 1,540 vs 1,545 chars (0.3%), document 835 vs 838 (0.4%), direct 84 vs 92 (9.5%, with benign *longer*).
- **The contamination guard mattered:** 31% of Gandalf prompts overlap the test set at ≥ 50% of word 8-grams, or match it exactly.
  Without the guard, training would have included near-copies of test injections.
- Tokens (real tokenizer): test indirect p50 214, p99 822, max 1,022. External p50 156, p99 612, max 768. None exceeds 1,024.
  7 long BIPIA test rows had their carrier trimmed to fit; every payload survived.
- 60 of 900 BIPIA email/table rows had lost their question in the v15 corpus; the rebuild restores it.

## v19b / v19b2 — built 2026-09-29 (not run)
- **v19b_extract.ipynb** (FRESH KERNEL, about 3–4 h plus about 30 min judge; resumable in 250-row shards). One forward pass per row
  for 9,299 rows: 5,916 external + 3,335 indirect test + a 48-row direct check. It stores the **end** and **anchor** states at 8 layers.
  The anchor is found from the tokenizer's character offsets. It **asserts that the anchor token never reaches into the untrusted data**
  (only trailing whitespace allowed: Llama-3 merges "?\n\n" into one token). It also asserts:
  (a) every direct row's anchor is the same (the causal prefix is identical);
  (b) the 48 re-extracted direct rows match the v14 cache at cosine ≥ 0.99, before the cache is reused for the 7,910 direct test rows.
  Then the 8B judge re-scores the 1,070 indirect rows of v17's subsample on the untrusted data (v17a2's prompt and code).
  **Disk: about 7 GB** (shards + assembled arrays).
- **v19b2_guards.ipynb** (any kernel, about 5 min): ProtectAI v2 and Prompt Guard 2 re-scored on the 3,335 rebuilt indirect rows,
  on the data (primary) and on task + data (secondary). Uses v17a's loader and positive-class probe. Direct rows keep their v17 scores.
- **Dry runs:** the cloud has no torch and cannot reach Hugging Face. So v19b ran against a numpy mock of torch and a causal mock Llama,
  using the real v19a CSVs, a Llama-3-style pre-tokenizer (special tokens split first; punctuation absorbs trailing newlines) and a
  mock cache. Anchors, gathering, shards, resume, the cache check, assembly and the judge path all passed.
  Real-Llama behaviour (MPS speed, the exact tokenization) is checked by the notebook's own asserts and smoke test.


## v19b / v19b2 — run complete (2026-09-29)
- Smoke test 0.23 rows/s, just above the 0.20 gate. The real run went at 0.3–6 rows/s per shard; the slowest stretch was
  shards 12–31, which hold the long email and document rows. All 38 shards finished.
- **Direct anchor constancy: cosine 1.00000 at every layer. Cache check: the 48 re-extracted direct rows match the v14 cache at
  cosine 1.00000 at every layer.** The cached end states are reused for the 7,910 direct test rows.
- Anchor examples are as intended: the anchor token is `' $ values paid.\n\n'` (the task's last token, which absorbs the newline),
  and the next token is the first word of the untrusted data.
- Judge: 1,070 rebuilt indirect rows scored on the data (0.7–2.1 rows/s).
- Guards (v19b2): ProtectAI's probe gap is 1.000 and PG2's is 0.967, with the positive class as in v17. On the rebuilt indirect rows,
  PG2's scores are tiny (mean 0.010 on injections vs 0.002 on benign). Only its ranking matters to the per-family thresholds.

## ADDENDUM — length confound (pre-registered 2026-09-29, BEFORE any MSLSAM, guard or judge v19 number was seen)

**What was found.** While dry-running v19c, the length-only control, computed on the **real** test texts, gave the following:

| family | injection median chars | benign median chars | length-only AUC | ASR of a pure length threshold @ 10% FPR |
|---|---|---|---|---|
| **SPML** | **501** | **76** | **0.994** | **1.2%** |
| DeepSet | 124 | 42 | 0.793 | 68.8% |
| BIPIA | 854 | 896 | 0.510 | 86.9% |
| InjecAgent | 399 | 388 | 0.546 | 87.1% |

On SPML the two classes' 5th–95th-percentile length ranges **do not even overlap** (injections p5 = 160 chars, benign p95 = 124).
**A length threshold alone "meets the target" on SPML.** Every SPML result in this project (v11–v18, including v16's 0.91% and
v18's 0.17%) must be read with this. The SPML benchmark cannot show that any detector understands injections.

**What changes in v19c (added before the primary method's numbers exist).** F1–F3 are still computed exactly as written.
**Additionally**, F1 and F3 are applied to the **length-clean families** only (length-only AUC ≤ 0.75, currently BIPIA and InjecAgent),
with the same 2-point margin and McNemar wins on ≥ half of those families with no losses. **Where the two readings disagree, the
length-clean reading governs the thesis claim.** F2 is already defined on BIPIA and InjecAgent. A descriptive length-overlap subset
is also reported: SPML has none; DeepSet has 177 injections and 181 benign in 35–196 chars.

**Disclosure.** The dry run used the real texts, so two *real* baseline results are already known: the length-only control above,
and TF-IDF trained on the external data (SPML 29.72 / DeepSet 39.12 / BIPIA 75.86 / InjecAgent 81.08 / pooled 45.97 ASR %). Every
MSLSAM, MiniLM, guard and judge number in the dry run was synthetic. No real MSLSAM-Δ number had been seen when this addendum was written.


## v19c — results (run 2026-09-29, ~25 min)

Every trained method saw **only** the 5,916 external rows. The inner CV picked C = 0.01 for every seed and both
MSLSAM variants, which is not a grid edge. Marginal cost of MSLSAM-Δ: **0.81 ms** on top of the 8B forward pass.

### ASR % at each family's own 10% FPR (MSLSAM: mean of 5 seeds [min, max])

| method | SPML ⚠length | DeepSet ⚠length | BIPIA | InjecAgent | pooled | indirect (BIPIA+InjecAgent) |
|---|---|---|---|---|---|---|
| **MSLSAM-Δ** (external only) | 11.7 [10.7, 13.3] | 14.5 [14.4, 14.7] | **61.8 [58.1, 63.9]** | 82.4 [80.7, 85.1] | 29.5 [27.8, 31.0] | 64.6 [61.2, 66.1] |
| MSLSAM-P-ext (external only) | 10.0 | **7.7** [6.3, 9.7] | 71.2 | 75.8 | 30.5 | 71.8 |
| TF-IDF + LR (external only) | 29.7 | 39.1 | 75.9 | 81.1 | 46.0 | 76.6 |
| MiniLM + LR (external only) | 56.6 | 68.4 | 87.2 | 71.4 | 66.9 | 85.1 |
| ProtectAI v2 [data] | **1.0** | 27.5 | 83.1 | 63.4 | 29.2 | 80.5 |
| Prompt Guard 2 [data] | 1.6 | 20.2 | 82.0 | 53.4 | **28.4** | 78.2 |
| 8B judge [data] (subsample) | 5.8 | 14.7 | **50.3** | **3.9** | **17.7** | **35.5** |
| length only (control) | **1.2** | 65.3 | 86.4 | 87.1 | 33.8 | 86.5 |

ROC-AUC on BIPIA: MSLSAM-Δ 0.81, PG2 0.62, **ProtectAI 0.42 (below chance)**. On InjecAgent: MSLSAM-Δ 0.66, the guards 0.77.
Hard-negative FPR (seed 0): MSLSAM-Δ 14.3% (BIPIA) / 14.5% (InjecAgent). PG2 11.0 / **26.8**, ProtectAI 7.3 / 8.1, TF-IDF 10.0 / 6.9.

### The pre-registered verdict, as fired

- **F1 (primary, all four families): MATCHES** the best full-coverage guard (PG2) in 4 of 5 seeds (LOSES in 1). Pooled 29.5 vs 28.4.
  **But** the pooled number is 60% SPML, which length alone solves; the guards get 1.0–1.6% there and MSLSAM-Δ gets 11.7%.
- **F1-LC (the governing reading, BIPIA + InjecAgent): LOSES in 5 of 5 seeds, by the rule.** Careful: this is the rule's residual
  category, not "worse overall". MSLSAM-Δ's indirect pooled ASR is **12–17 points LOWER (better)** than PG2's, and McNemar favours it
  on BIPIA in every seed. The guard is significantly better on InjecAgent in every seed, and BEATS requires no losses. The honest
  description is **complementary**: MSLSAM-Δ is far better on documents and email (BIPIA 58–64% vs 82–83%), and far worse on agent tool
  output (InjecAgent 81–85% vs 53–63%). InjecAgent's tool-output format has no analogue in the external training data, which
  was stated as a limitation in advance.
- **F2 (does the task-drift Δ help indirect injection?): SEED-DEPENDENT (2/5).** Δ improves indirect pooled ASR over the last token by
  **5.7–10.8 points in every seed** (BIPIA 71 → 62). But it is worse on InjecAgent (82 vs 76), and McNemar says so significantly in 3 seeds.
- **F3 (hidden states vs text under identical external training): SEED-DEPENDENT (2/5).** Against **TF-IDF it passes in 5/5**
  (15–18 pooled points, wins on SPML, DeepSet and BIPIA, no losses). Against **MiniLM** it wins by 36–39 points pooled, but MiniLM is
  significantly better on InjecAgent in 3 seeds. F3-LC: 2/5.
- **Target (ASR < 5% on the worst seed): not met on any family** under fair training.
- **The 8B judge** (a separate forward pass, 378 ms) is the strongest detector on indirect injection: InjecAgent 3.9%, BIPIA 50.3%.
  Against MSLSAM-Δ, McNemar gives the judge InjecAgent (119 vs 1) and SPML, and ties BIPIA and DeepSet.

### What this means (read with the length addendum)

1. **Under a fair protocol, MSLSAM is competitive with purpose-trained guards but does not beat them.** It is not a
   drop-in replacement. Its errors are **complementary** to theirs: it catches document and email injections the guards miss,
   and misses tool-output injections they catch.
2. **The task-drift Δ does what it was designed for on documents** (BIPIA −9 points versus the last token, and the best BIPIA
   result of any cheap detector). It does not transfer to a format it never saw (InjecAgent).
3. **DeepSet, once length is controlled**, still separates well. In the length-overlap subset (177 injections, 181 benign, 35–196 chars),
   MSLSAM-Δ reaches 8.05% against the guards' 20–27% and length alone at 84%. This is descriptive, not pre-registered as a claim.
4. **Hidden states beat a text classifier trained on the same external data** (TF-IDF, 5/5 seeds). That confirms v18/v18b's H2 outside the corpus.
