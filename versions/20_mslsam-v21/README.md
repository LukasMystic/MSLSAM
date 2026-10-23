# 20 — MSLSAM v21 (confirmation on fresh data + the inference mode + time and space complexity)

**Status:** ✅ COMPLETE 2026-10-18 — **C1 BEATS (5/5) · C2 does not hold (0/5)**; target reached only by the ensemble on Sum-TT. Llama benchmark abandoned (disclosed). v21a done (I1 part 1 passed). v21b run 1 died after the extraction, so v21b was split into v21b (Llama) + v21b2 (guards), 2026-09-30. **The pre-registration below is fixed now, before any v21 number exists.**

## Why

v20b's headline was that MSLSAM-Δ, trained on external data plus tool outputs, **beats Prompt Guard 2 on the length-clean indirect families**
(BIPIA + InjecAgent). The result held in 5/5 seeds: pooled ASR 63.9 vs 78.2.

That verdict was pre-registered, but the design behind it was not blind. The tool-output data, the Δ feature and the choice of
comparator were all chosen after v17–v20a had looked at the **same four test families**. The comparison may therefore be overfitted to that
test set. The thesis needs to answer two things before it can make the claim:

1. **Does the frozen v20b detector keep its result on data that no version has touched?**
2. **How does the defence actually run, and what does it cost?** This covers MSLSAM on its own and the MSLSAM + Prompt Guard 2 ensemble,
   and asks for time and memory numbers, not just ASR.

## Design

### The frozen detector (v21a)
- MSLSAM-Δ v20b is **refitted, not changed**, for seeds 0–4: the same data (v19 external + v20 tool, 7,416 rows), code and C (inner CV).
- **Integrity I1:** every refit must reproduce v20b's saved test scores (max |diff| < 1e-4). Otherwise the notebook stops.
- The fitted pieces are saved to `artifacts/mslsam_v20b_s{seed}.npz`. That file holds, per layer, the PCA mean and components, then the scaler
  and the LR weights. A **numpy-only head** (no sklearn) must reproduce sklearn's scores (< 1e-4). The artifact is what a deployment loads.

### The fresh test set (v21a): BIPIA-style summarization of news articles, never used in this project
Nothing in v21's test data was used in v14–v20: not the carriers, not the task, and for one family not the payloads either.

| family | injections | clean (benign) | hard negative | payload source |
|---|---|---|---|---|
| **Sum-BIPIA** | 300 | 300 | 300 | BIPIA's 150 text attacks × 2 carriers. **The attack strings were in the v19 test set.** Carriers and task are new |
| **Sum-TT** | 300 | 300 | 300 | 150 **Tensor Trust** human-written hijacking attacks (GitHub `HumanCompatibleAI/tensor-trust-data`). **Never used anywhere** in this project |

- **Carriers:** 1,000 XSum BBC articles, selected by BIPIA's own `abstract/index.json`. Each carrier is used once (600 carriers), and the
  other 400 articles donate the clean filler.
- **Task:** "Summarize the following news article in one sentence." No v14–v20 row used a summarization task.
- **Three tiers per carrier,** with the same article text, the same position (start / middle / end, drawn at random) and length matching to the payload:
  - injected
  - clean: declarative sentences from another article
  - hard negative: a benign Dolly instruction
- **Data length ≤ 1,800 characters,** so that the untrusted data fits Prompt Guard 2's 512-token window. v21b counts the rows over 512 tokens
  and scores any over-length row as the maximum over 512-token windows.
- **Tensor Trust filter:**
  - ASCII-clean text of 40–600 characters, with at least 6 distinct words
  - near-duplicates removed
  - contamination guard against the training data: ≥ 50% of word 8-grams, or an exact match for short texts
  - then 150 attacks drawn with seed 42
- **Every row** is checked with the real Llama tokenizer and must fit 1,024 tokens.

### Methods scored on the fresh data (v21b, v21c)
| method | role |
|---|---|
| **MSLSAM-Δ v20b (frozen, seeds 0–4)** | primary |
| Prompt Guard 2 86M, on the untrusted data | the comparator: v19's best full-coverage guard, the same one v20b beat |
| **Ensemble z-mean** = (z_Δ + z_PG2) / 2 | the ensemble. z = (logit − μ) / σ, where μ and σ come from **external + tool benign** rows (MSLSAM: out-of-fold scores). No fresh row is used |
| ProtectAI v2, the Llama-3-8B judge, length only | descriptive references |

**Why z-mean and not v20a's z-max.** v20a's primary z-max failed because MSLSAM-Δ sat about +3.6 SD above its reference on every tool row: the
external-benign reference did not transfer, and a max is dominated by whichever score has drifted. A mean does not have this problem. With per-family
thresholds, μ drops out entirely, and the mean is a fixed weighting 1/σ_Δ : 1/σ_PG2 of the two logits. z-mean was v20a's
**secondary** result. It was never claimable there. v21 is its first pre-registered test, on data it has not seen.

### Evaluation (v21c): the v17b protocol, unchanged
- Each fresh family is scored at its own 10% FPR budget on its **clean** tier, using nested thresholds (400 resamples, tie-aware).
- **FRESH pooled** is the injection-weighted ASR over Sum-BIPIA and Sum-TT, each at its own threshold.
- McNemar tests are paired on identical injections, p < 0.05.
- MSLSAM runs with seeds 0–4.

## Pre-registered reading (fixed 2026-09-29, before any v21 number exists)

- **I1 (integrity, checked first; a failure stops v21):**
  - the refits reproduce v20b (< 1e-4)
  - the numpy head reproduces sklearn (< 1e-4)
  - the deployment path gives the same score as the batched extraction on the 50 benchmark prompts: batch 1, `use_cache=True`, KV cache kept for
    decoding, |Δp| ≤ 0.02
- **C1: does v20b's result replicate?** Compare MSLSAM-Δ with Prompt Guard 2 on FRESH pooled ASR.
  - **BEATS** if Δ is ≥ 2 points lower, with a McNemar win on ≥ 1 fresh family and a loss on none.
  - **MATCHES** if the difference is under 2 points.
  - **LOSES** otherwise.
  - The verdict must hold in **≥ 4 of 5 seeds**; otherwise it is *seed-dependent*.
  - If C1 is BEATS, v20b's T3 is **confirmed on unseen data**. If it is MATCHES, the thesis claims parity at a fraction of the cost.
    If it is LOSES, the thesis states plainly that **v20b's T3 did not replicate** and was specific to the test set.
- **C2: does the ensemble earn its cost?** z-mean must meet all of the following, in **≥ 4 of 5 seeds**:
  - FRESH pooled ASR ≥ 2 points below the **better** of its two components
  - no McNemar loss against either component on either family
  - a McNemar win against at least one component on at least one family
- **Target (ASR < 5% at FPR ≤ 10%):** per fresh family, for MSLSAM-Δ and the ensemble. It is claimed only if the **worst** seed meets it.
- **Length control:** if length-only AUC (injected vs clean) is > 0.60 in either direction on a fresh family, that family's comparison is flagged.
  The fresh rows are built length-matched, so the bar is stricter than v19's 0.75.
- **Descriptive only:**
  - Sum-BIPIA vs Sum-TT separately. Sum-TT is the cleaner test, because its payloads are new too.
  - hard-negative FPR
  - ASR by position
  - ProtectAI, the 8B judge
  - **deployment thresholds.** Each method's threshold is fixed at 10% FPR on the external + tool benign rows, with no fresh calibration.
    v21c reports the resulting FPR and ASR on the fresh families: what a deployment would see without in-domain calibration.

**Cannot be changed after the run:** the fresh data, the frozen artifacts, the ensemble rule, the thresholds, the families, the seeds.
**v21 is the last look at the fresh families. Any later version must build new fresh data.**

## Inference mode (what a deployment runs) and how v21b measures it

```
app assembles  [system prompt] [task] [untrusted data]        (chat template, one string)
      │
      ▼
Llama-3-8B prefill, ONE pass, use_cache=True, 8 forward hooks
      │   hooks copy h_anchor (last task token) and h_end (last prompt token) at 8 layers: 2·8·4096 fp16 = 128 KB, one copy to the CPU
      ├──────────────► MSLSAM head (numpy, CPU): Δ = h_end − h_anchor → 8× PCA(256) → scaler → LR → p_Δ
      │                                   [ensemble only] Prompt Guard 2 on the untrusted data → p_PG2 (separate 86M model)
      │                                   combine: z-mean → compare with the per-context threshold
      ▼
flag?  yes → block / ask the user;   no → decode the answer from the SAME KV cache (no second prefill)
```

- **Latency B1:** batch 1, warm, 50 fresh prompts stratified by tier and length. Each stage is timed separately with device synchronisation:
  - tokenize and anchor
  - prefill **without** hooks (the undefended model)
  - prefill **with** hooks
  - MSLSAM head
  - Prompt Guard 2
  - combine
  - the 8B judge (a second prefill)
  - and, for context, decoding 32 tokens from the kept cache

  Medians and p90 are reported.
- **Scaling B2:** a length sweep at n = 128 / 256 / 512 / 1,024 prompt tokens, measuring prefill and head time against n.
- **Memory B3:**
  - parameter bytes (Llama, Prompt Guard 2, ProtectAI)
  - device memory after loading each model
  - KV-cache bytes per token, measured
  - hook buffers
  - artifact file sizes

### Complexity (the thesis table; v21c fills in the measured column)
| component | time | extra memory | shared with the LLM? |
|---|---|---|---|
| Llama prefill (n tokens, L = 32 layers, d = 4096) | O(L·(n·d² + n²·d)) | weights 8.0 B params; KV cache O(L·n·d_kv) | **yes**: the app pays it anyway |
| hooks (K = 8 layers, 2 positions) | O(K·d) copy | O(K·d) = 128 KB (fp16) | the same pass |
| MSLSAM head: K PCA projections + LR | O(K·d·k) = 8·4096·256 ≈ 8.4 M MACs, **constant in n** | O(K·d·k) ≈ 8.4 M params (33.6 MB fp32) | no, but it is negligible |
| Prompt Guard 2 (m data tokens, ≤ 512 per window) | O(⌈m/512⌉ · L_g·(512·d_g² + 512²·d_g)) | ≈ 280 M params (86 M backbone + embeddings; measured in v21b) | **no**: a second model |
| ensemble | sequential: sum; parallel: max(Llama+head, PG2) | Llama + PG2 | — |
| 8B judge | a **second** Llama prefill over the data + judge prompt | its own KV cache | no |

## Run order (revised 2026-09-30 after v21b run 1)
1. **v21a_freeze_build.ipynb** (any kernel, CPU). ✅ DONE.
2. **v21b_extract_infer.ipynb** (FRESH kernel, MPS, loads **only Llama**, ~30–40 min). The extraction is already done and is skipped.
   The notebook runs B1 and B2 on the Llama side (saved after every prompt, resumable), then the judge on all fresh rows.
3. **Kernel → Restart**, then **v21b2_guards.ipynb** (no Llama, ~10–15 min). It covers:
   - the guard side of B1 and B2, on the same prompts
   - Prompt Guard 2 and ProtectAI on the fresh rows, plus Prompt Guard 2 on external + tool (the ensemble reference)
   - the merge into `bench.json`
4. **v21c_confirm.ipynb** (any kernel, CPU, minutes). It applies I1, C1, C2, the target and the length flag, then adds the deployment thresholds
   and the complexity table.

## Run log
- **v21a (2026-09-29): OK.**
  - XSum md5s **match BIPIA's** for both train and test, so the articles are exactly BIPIA's.
  - Tensor Trust: 194 attacks usable, 150 drawn; none overlap training or the v19 test set.
  - 877 of 1,000 articles are usable (at least 600 characters and not contaminated), giving 600 carriers and 277 donors.
  - Length-only AUC: 0.500 for Sum-BIPIA, 0.506 for Sum-TT, so both are length-clean.
  - Prompt tokens: median 397, max 729.
  - **I1 part 1 PASSED:** every seed reproduces v20b to 3.0e-08, and the numpy head matches sklearn to 3.7e-06.
  - Reference: OOF AUC 0.997 (0.998 on tool rows); benign logit μ −4.11, σ 1.93–1.94; external 10%-FPR threshold p ≥ 0.148–0.150.
  - The many "divide by zero / overflow in matmul" RuntimeWarnings are macOS Accelerate's known **spurious** warnings. The 3e-08 reproduction
    proves the numbers are finite and exact, and v21c now asserts that every score is finite.
- **v21b run 1 (2026-09-29/30): kernel died after the extraction.**
  - All 8 shards and the fresh arrays are saved.
  - The death came somewhere in the benchmark cell or the guard-loading cell. Nothing was printed and nothing from B1 was written.
  - The likely cause is memory: Llama fp16 (≈ 16 GB) plus two fp32 guards (≈ 1.9 GB) plus the editor, on 24 GB.
  - Shards 0–2 took about 1 h each; after a restart, shards 3–7 took about 4 min each. So the first session was already swapping.
  - **Fix:** Llama and the guards now run in separate kernels. B1 is saved after every prompt, and every stage skips finished work.
  - Consequence (disclosed): the two sides are timed in separate processes. Sequential ensemble latency = the sum; contention is not measured.
- **v21b run 2 (2026-09-30): stuck for 2 h in B1's warm-up** ("B1: 0 prompts already timed, 50 to do") with no output. There is no
  traceback, so the stuck line is unknown. The most likely cause is swapping: memory pressure makes every step page the 16 GB of weights back
  in, so the run crawls instead of crashing. That is consistent with run 1's slow shards. **Fix (run 3):**
  - a pre-flight on one prompt that prints each stage and memory/swap, and stops if a single step takes more than 10 s
  - a per-prompt progress line
  - a heartbeat file, `results/v21/_b1_heartbeat.txt`
  - warm-up cut from 3 to 2
- **v21b run 3 (2026-09-30 → 10-10): ABANDONED.** Its own memory report showed **swap 50.1 GB used of 51.2 GB**. The extraction smoke test ran at
  0.20 rows/s, below the gate. The pre-flight prefill took 2.5 s for 282 tokens, then hung in the 4-token decode (heartbeat:
  "row 56 stage decode 4 tokens STARTED"). In that machine state, latency measures swapping, not the method.
  **Decision (2026-10-10):**
  - The Llama-side benchmark is dropped.
  - The confirmation (C1/C2) does not need Llama, because the fresh states came from run 1.
  - **v21b2_guards.ipynb** is now standalone: guard scores, the ensemble reference, and guard latency/memory.
  - **v21c** builds the cost table from the head timed on the frozen artifact (CPU), Prompt Guard 2 from v21b2, and the Llama prefill (326 ms)
    and judge (378.5 ms) from v17's healthy measurements, cited.
  - **Deviations, disclosed:**
    - I1 part 3 (deployment path = batched extraction) is NOT RUN.
    - The judge on the fresh rows is NOT RUN.
    - The hook overhead, decode-from-cache and Llama length sweep are not measured; theory only.
- **Run order now:** restart the Mac (clears swap) → v21b2_guards (fresh kernel) → v21c_confirm.
- **Recap (2026-10-11):** `Experiments/outputs/v21_recap.md`.
- **v21b2 run 1 (2026-10-18): `MPS backend out of memory`** (22.1 GB allocated) while scoring the first job. The guards themselves take 1.9 GB.
  Cause: every batch was padded to its own exact length, so the MPS cache kept buffers for hundreds of shapes. Nothing was saved.
  **Fix:** pad batches to a multiple of 64 (≤ 8 shapes, as v19b2 did), batch 32 → 16, `torch.mps.empty_cache()` every 10 batches, and
  progress lines on long jobs. Padding is masked, so the scores are unchanged. Swap at the start was still 6.2 GB after a restart.
  The run-1 notebook is in `_to_delete/`.

## v21 RESULT (run 2026-10-18) — C1 BEATS (5/5) · C2 does not hold (0/5) · target reached only by the ensemble on Sum-TT

ASR % at each fresh family's own 10% FPR (clean tier). Mean of 5 seeds; range in brackets.

| method | Sum-BIPIA | Sum-TT (payloads never used) | FRESH pooled | hard-neg FPR BIPIA / TT |
|---|---|---|---|---|
| **MSLSAM-Δ v20b (frozen)** | **17.7** [16.6–19.6] | **5.5** [5.2–5.8] | **11.6** [11.2–12.6] | 37 / 71 |
| ensemble z-mean | 19.8 [18.4–20.8] | **3.1** [2.7–3.3] | 11.4 [10.8–12.0] | 29 / 65 |
| Prompt Guard 2 | 67.3 | 8.8 | 38.1 | 10 / 14 |
| ProtectAI v2 | 75.1 | 4.0 | 39.5 | 10 / 12 |
| length only | 89.7 | 89.9 | 89.8 | — |

- **I1:** parts 1–2 passed (the refit equals v20b to 3e-08; the numpy head equals sklearn to 3.7e-06). Part 3 was NOT RUN (disclosed).
- **C1 — BEATS in 5/5 seeds.**
  - MSLSAM-Δ is 25–27 points below PG2 on FRESH pooled.
  - McNemar: wins on Sum-BIPIA in every seed (149–156 vs 4–8 discordant). Sum-TT is not significant (21–23 vs 10–12, p 0.06–0.12). No losses.
  - **v20b's T3 is confirmed on data no version had seen.**
- **C2 — does not hold (0/5).**
  - The ensemble is −0.07 to +0.63 points vs MSLSAM pooled. Seed 2 has a McNemar loss on Sum-BIPIA.
  - Post-hoc and descriptive: the ensemble helps on Sum-TT (significant in 4/5 seeds, 7–9 vs 0–1) and slightly hurts Sum-BIPIA. The two effects cancel when pooled.
- **Target (pre-registered rule: worst seed ASR < 5% at FPR ≤ 10%, protocol tolerance 10.5):**
  - Met only by the **ensemble on Sum-TT**: worst seed 3.33%. Wilson upper bound 6.0%; realised FPR 10.4–10.5%.
  - MSLSAM-Δ alone on Sum-TT: worst seed 5.80%, so not met.
  - Borderline: the point estimate is under 5%, but the CI crosses 5% and the FPR is 10.5%, not strictly < 10%.
- **Length:** clean (AUC 0.500 / 0.506).
- **Position (seed 0):** start 17.9, middle 13.5, end 7.1. End payloads are easiest, consistent with reading the end-of-prompt state.

### The costs and caveats that must be stated with C1
1. **Hard negatives.** MSLSAM-Δ flags harmless instructions inside articles 34–39% (BIPIA tier) and **70–74%** (TT tier) of the time. PG2: 10–14%.
   MSLSAM detects *that an instruction-like drift is present*, and it does not tell a benign instruction from a malicious one. Part of its win on
   Sum-BIPIA comes from BIPIA's own definition: many BIPIA "attacks" are benign-looking questions ("What is the capital of Brazil?"), which PG2 lets through by design.
2. **Calibration does not transfer.** With thresholds set on external benign rows only (no fresh calibration), MSLSAM-Δ's FPR on clean news is
   **28–71%**, and it varies a lot by seed. The ensemble's is 5–19%; PG2's is 0.7% (with ASR 95 / 21). Per-context calibration is a required deployment step.
3. **The fresh families look easier than v19's BIPIA.** There is one clear task and a long news carrier: MSLSAM 17.7 here vs 61.8–65.2 on v19 BIPIA, and PG2 8.8 on Sum-TT.
   The comparison is fair (same data for every method), but absolute ASR does not transfer across benchmarks.
4. **Cost (batch 1):**
   - MSLSAM head: 1.37 ms (numpy, CPU, v21c)
   - PG2: 76 ms median on the fresh data (v21b2; 279 M params, 1.12 GB)
   - ensemble sequential: +23.8% time to first token; in parallel, about the MSLSAM cost alone
   - Llama prefill 326 ms and judge 378 ms (both cited from v17)
