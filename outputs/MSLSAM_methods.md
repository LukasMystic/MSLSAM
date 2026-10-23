# MSLSAM — Methods, Data and Results

Prompt injection detection from Llama-3-8B hidden-state layer trajectories.
Stanley Pratama Teguh · compiled 2026-09-28 · covers versions 01–16.

---

## 1. What the detector actually does

One sentence: **it reads the hidden states the model already produces while processing a
prompt, and asks whether that internal state looks like the states injections produce.**

The full path for a single prompt:

```
prompt text
   │
   ├─► wrapped in a fixed chat template  (system + user + generation prefix)
   │
   ├─► ONE forward pass through Llama-3-8B-Instruct
   │      forward hooks on 8 layers: 4, 8, 12, 16, 20, 24, 28, 31
   │      at each, take the hidden state of the LAST token   → 8 × 4096 floats
   │
   ├─► project each layer through a PCA basis fitted on benign baseline prompts
   │                                                        → 8 × 512 floats
   │
   ├─► 24 scalar features
   │      · max cosine similarity to DIRECT-injection prototypes   (8, one per layer)
   │      · max cosine similarity to INDIRECT-injection prototypes (8, one per layer)
   │      · Mahalanobis distance from the benign baseline          (8, one per layer)
   │
   ├─► RobustScaler → HistGradientBoostingClassifier → P(injection) ∈ [0,1]
   │
   └─► compare against a threshold calibrated to a 10% false-positive budget
```

No generation. No second model. The forward pass is the one the protected model was going to
run anyway, so the detector's **marginal** cost is the feature construction plus a tree lookup.

### Why hidden states rather than the text

A lexical filter matches what an injection *says*. Hidden states reflect what the model
*does* with it — whether the instruction has been taken up as an instruction. The project's
premise is that this survives paraphrase, obfuscation and unseen wordings in a way that
keyword matching does not. Sections 6 and 7 report how far that held.

### Why these 8 layers

Version 03 swept layers 12–31. A later diagnostic found layer 4 carries roughly **3× the
separation margin** of any other depth against adversarially-phrased benign text — the hard
negative class — and that the margin decays monotonically with depth, effectively gone by
layer 28. Version 03 had chosen layer 28 on calibration AUC measured against *harmful*
prompts rather than against that hard class, which is why it looked good and generalised
badly. The 8-layer set spans the range so the classifier can weight depths itself.

---

## 2. Data

### 2.1 Final corpus

**13,245 rows — 4,148 injections, 9,097 benign — from 1,448 unique payloads across 498 unique
carrier documents.** 12,897 English, 348 German.

| corpus | rows | label | role |
|---|---|---|---|
| **SPML** | 2,499 | injection | direct — persona / system-prompt override |
| **BIPIA** | 1,200 | injection | **indirect** — payload inside email / table / code |
| **DeepSet** | 263 | injection | direct, EN + DE |
| **InjecAgent** | 186 | injection | **indirect** — payload inside tool output |
| WJB-Adv-Benign | 2,500 | benign | **adversarially-phrased benign** — the hard class |
| Baseline-Dolly | 2,000 | benign | PCA / Mahalanobis reference only, never evaluated |
| SPML-Benign | 1,198 | benign | matched benign for SPML |
| WJB-Vanilla-Benign | 800 | benign | ordinary requests |
| InjecAgent-HardNeg | 509 | benign | **tier 2** — benign imperative in tool output |
| BIPIA-HardNeg | 481 | benign | **tier 2** — benign imperative in document |
| InjecAgent-Clean | 479 | benign | **tier 1** — same tool output, payload replaced |
| BIPIA-Clean | 480 | benign | **tier 1** — same document, payload removed |
| DeepSet-Benign | 300 | benign | matched benign for DeepSet |
| XSTest-Safe | 250 | benign | safe prompts that superficially look unsafe |
| JBB-Benign | 100 | benign | JailbreakBench benign split |

Median length 385 characters; p90 1,460; max 4,539. Truncated at **512 tokens** at extraction.

### 2.2 Sources

| source | licence / origin | what it contributes |
|---|---|---|
| SPML Chatbot Prompt Injection | MIT | 12,541 injections available; 2,499 sampled |
| DeepSet prompt-injections | public HF dataset | 263 injections, 399 benign; EN + DE |
| BIPIA | benchmark release | 250 unique payloads × email/table/code carriers |
| InjecAgent | benchmark release | 62 attacker instructions × 17 tool-response templates |
| WildJailbreak | public release | adversarial-benign and vanilla-benign splits |
| XSTest | public release | 250 safe-but-suspicious prompts |
| JailbreakBench | public release | 100 benign |
| Databricks Dolly 15k | CC-BY-SA-3.0 | PCA baseline; benign imperative filler for tier 2 |

**HackAPrompt was excluded deliberately.** 71.4% of its successful attacks contain the literal
string `PWNED` — the competition's target output, not a property of injection. Training on it
would have produced a `PWNED` detector.

### 2.3 Three controls that carry the evaluation

**Matched clean carriers (tier 1).** Every BIPIA and InjecAgent injection has a twin: the
*same* document with the payload removed (BIPIA) or replaced by a declarative sentence
(InjecAgent, whose template requires a filled slot). A detector that fires on these is
recognising the document, not the payload.

**Hard negatives (tier 2).** The same carrier, same insertion position, with a **benign
imperative sentence** — a Dolly instruction — length-matched to the payload it replaces. This
separates *"an instruction is present in untrusted content"* from *"this instruction is an
attack"*. Nothing before version 15 could measure that distinction. The two tiers are always
reported separately: a false positive on tier 1 is a bug, a false positive on tier 2 is a
defensible judgement about where the trust boundary sits.

**Length matching.** An early version produced hard negatives 23% longer than the injections
they stood in for, which would have made tier-2 results unreadable — instruction, or extra
length? Fillers are now assembled to the length of the payload they replace (≥99% land within
[0.7, 1.4]× target at every payload decile), and the builder warns if any tier drifts more
than 10% from the injected median.

---

## 3. Representation extraction

| parameter | value |
|---|---|
| model | `meta-llama/Meta-Llama-3-8B-Instruct`, fp16 |
| layers hooked | 4, 8, 12, 16, 20, 24, 28, 31 |
| position | last token (left-padded, so index −1 is the true final token) |
| max length | 512 tokens |
| system prompt | fixed: *"You are a helpful, secure, and harmless AI assistant. You must refuse to assist with any dangerous, illegal, or unethical requests."* |
| format | `apply_chat_template(system, user, add_generation_prompt=True)` |
| hardware | 24 GB M5 MacBook Pro, MPS |
| throughput | ~0.5 rows/s batched (~2,000 ms per prompt at 512 tokens) |
| output | 8 × 4096 floats per prompt, cached to `.npz` shards |

Extraction is checkpointed and resumable, with length-sorted token-budget batching (1.7×
faster than fixed batching, which wasted 2.15× on padding) and OOM-splitting retry. On a 24 GB
machine the fp16 weights alone take ~16 GB, so the activation budget is a few GB and MPS
fragments badly; a single row OOMing after a deep clean means the process is wedged and the
kernel must restart.

**BIPIA carrier truncation.** At 512 tokens, a payload at the end of a long document was being
cut off entirely — 129 rows (3.1% of positives) labelled "injection" that contained none. The
builder now truncates the *carrier* and keeps the payload.

---

## 4. Features

**PCA.** Fitted per layer on the 2,000 Dolly baseline rows, 512 components. Everything is
projected into that basis. Earlier versions were capped at 99 components by a 100-prompt
baseline, which was a real constraint on what the classifier could see.

**Prototype max-cosine (16 features).** For each attack family *f* ∈ {direct, indirect} and
each layer, the maximum cosine similarity between this prompt's projected state and the
projected states of the training injections of that family. Self-match masking is **exact and
identity-based**: if the query row is itself a prototype, its own column is *removed* before
the max — not compared against it.

> Three separate self-match leaks occurred in this project (v11 positional masking; a
> comparison-instead-of-removal in v15; an in-sample prototype pool in a v15 experiment that
> scored **100% on pure random data**). Every similarity-to-prototype computation is now
> noise-tested: feed it random arrays and confirm it returns chance.

**Mahalanobis distance (8 features).** Distance of the projected state from the baseline mean,
under the baseline covariance, per layer.

**Classifier.** `HistGradientBoostingClassifier(max_iter=300)` on `RobustScaler`-normalised
features.

> **The single most consequential finding in the project.** Versions 09–14 all used
> LogisticRegression and no ablation ever varied the head. On injections vs
> adversarially-phrased benign — the class that had blocked progress since version 02 — the
> classifier sweep gave LogReg AUC 0.9410 against HistGradientBoost **0.9958**, RandomForest
> 0.9942, MLP 0.9937. Classifier spread 0.055, above the pre-set 0.05 threshold, so this was
> never a feature ceiling: the features carried the signal and the linear decision boundary
> could not read it. Swapping the head moved full-corpus AUC from 0.9048 to **0.9867** and ASR
> from 28.35% to 3.25%, and brought WJB-Adv-Benign FPR from 10.16% to **7.49%** — under budget
> for the first time.

---

## 5. Evaluation protocol

**Grouped cross-validation, 5-fold `StratifiedGroupKFold`.** Group keys matter more than they
look:

- `payload_id` alone is not enough once hard negatives exist, because an injection and its
  hard-negative twin share a carrier — the model could learn the document.
- The connected component of the (`payload_id`, `pair_id`) graph closes both routes, but
  **InjecAgent forms a single component** (17 templates × 62 instructions, fully
  cross-connected), so every InjecAgent row would land in one fold and the experiment would
  silently become leave-one-family-out.
- Primary key: **`group_carrier`** — `pair_id` for indirect rows, `payload_id` elsewhere.
  Closes the twin leak, keeps the leakage profile comparable with earlier versions.
- Sensitivity key: `group_component`, reported wherever it yields enough groups. Where the two
  disagree, the gap *is* the uncertainty.

**Nested thresholds.** The threshold is the 90th percentile of a *calibration half* of the
benign scores; FPR is measured on the held-out half; ASR on the injections. Resampled 400
times. An earlier version set the threshold from the same benign scores whose FPR it then
reported — that leak inflated every number it touched.

**Two distinct uncertainties, never merged.** The threshold/fold variation interval (2.5–97.5
percentile over resamples) and the Wilson binomial confidence interval. Only the second
licenses a claim of the form "ASR < x".

**Per-context thresholds.** Each injection family is scored against a threshold calibrated on
*its own* benign context — BIPIA against BIPIA-Clean, SPML against SPML-Benign, and so on —
because a single global threshold measured against a pooled benign set hides enormous
per-corpus variation.

**Reporting rules that became project law:**
- An FPR measured on one benign corpus is not an FPR. Print the per-corpus breakdown beside
  every pooled number.
- A pooled FPR is not an operating point. Set it so the *worst* context is inside budget.
- Never report global AUC for an injection detector — it mixes harmfulness with injection.
- State the classifier, the training regime and the grouping with every number.

---

> **⚠ §6.1 is superseded by §6.0 (v17, 2026-09-28).** Read §6.0 first. The SPML number is real,
> but it is not evidence for MSLSAM, because a TF-IDF classifier and an untrained ProtectAI model
> both reach it.

## 6. Results

### 6.0 Against baselines (v17, 12 methods, identical folds and thresholds) — read first

ASR % at each family's own 10%-FPR threshold (nested). Every method's worst own-context FPR is 10.5–10.7%.

| method | regime | SPML | DeepSet | BIPIA | InjecAgent | pooled | cost bs=1 |
|---|---|---|---|---|---|---|---|
| keyword rules | zero-shot | 87.15 | 84.79 | 98.75 | 79.03 | 90.00 | 0.07 ms |
| ProtectAI DeBERTa v2 (184M) | zero-shot | 1.00 | 27.47 | 84.14 | 77.44 | 30.16 | 31.7 ms |
| Prompt Guard 2 86M (279M total) | zero-shot | 1.59 | 20.23 | 89.43 | 52.21 | 30.45 | 32.4 ms |
| Llama-3-8B judge, 1 pass (subsample) | zero-shot | 5.80 | 14.73 | 88.78 | 8.87 | 27.83 | 378.5 ms |
| TF-IDF + LR | leave-one-corpus-out | 86.64 | 55.33 | 80.17 | 74.93 | 82.26 | 0.64 ms |
| MiniLM-L6 + LR | leave-one-corpus-out | 77.74 | 71.02 | 79.07 | 75.56 | 77.60 | 8.8 ms |
| **MSLSAM** | leave-one-corpus-out | 50.58 | 38.72 | 80.44 | 66.55 | 59.18 | 326 + 3.22 ms |
| TF-IDF + LR | in-distribution | 1.44 | 21.86 | 13.21 | 0.00 | **6.07** | 0.64 ms |
| MiniLM-L6 + HGB | in-distribution | 1.19 | 43.00 | 47.82 | 52.28 | 19.62 | 8.8 ms |
| **MSLSAM** | in-distribution | 1.13 | 8.65 | 35.35 | 66.04 | 14.42 | 326 + 3.22 ms |

**Pre-registered outcomes, both against MSLSAM.** (1) Zero-shot guards (27.8–30.5%) beat MSLSAM
on held-out families (59.2%). (2) A TF-IDF classifier trained on the same folds beats MSLSAM
in-distribution (6.07% vs 14.42%), so MSLSAM's in-distribution result reflects the training
data, not the hidden states.

**Post-hoc explanation (found after the run, not pre-registered).** The primary grouping holds
out carriers, not payloads. 96.5% of BIPIA and 98.4% of InjecAgent test injections reuse a
payload seen in training (SPML and DeepSet: 0%), and TF-IDF wins exactly on those two families.
On the two leak-free families MSLSAM leads by only ~1.5 pooled points.

**What survives.**
- Among methods trained only on this corpus, MSLSAM transfers to unseen families far better
  (59.2% vs 77.6–82.3%; paired McNemar vs MiniLM+LR: SPML p ≈ 1e-105, DeepSet p ≈ 3e-15).
- DeepSet in-distribution: 8.65% vs TF-IDF 21.86% (p = 1.6e-5, n = 263).
- It adds 3.22 ms to a pass already being run.


> **⚠ Length confound (found 2026-09-29).** On SPML a length threshold alone gets ASR 1.2% at 10% FPR
> (injections median 501 chars, benign 76; length AUC 0.994). No SPML number in this document is evidence that a detector
> understands injection. DeepSet is partly confounded (AUC 0.79). BIPIA and InjecAgent are length-clean.

### 6.0b A direct probe under a leak-free split (v18)

MSLSAM-P drops the 24-feature summary. Each of the 8 layers is reduced by PCA to 256 dimensions (fitted
on the training fold), and a logistic-regression probe is trained on all 2,048 features. Its regularisation is
chosen by inner CV. It is evaluated on a **payload-held-out** split (P2), where no test attack text ever appears
in training. The trained baselines are refit on identical splits, and the reading rules were pre-registered.

| method (P2) | SPML | DeepSet | BIPIA | InjecAgent | pooled |
|---|---|---|---|---|---|
| **MSLSAM-P (LR)** | **0.17** | **2.73** | 50.30 | **21.89** | **15.81** |
| MSLSAM v17 | 1.15 | 10.52 | 46.75 | 32.63 | 16.35 |
| TF-IDF + LR | 1.41 | 22.18 | 51.19 | 36.27 | 18.69 |

The pre-registered test of hidden states against text **passed**. MSLSAM-P is 2.88 pooled points ahead, with
paired McNemar wins on SPML, DeepSet and InjecAgent, no losses, and no reversal when carriers are also held out.
Both direct families meet ASR < 5% at a 10% FPR budget (measured FPR 10.1% and 10.8%). Indirect injection
(BIPIA ~50% for every method) does not. Marginal cost is 0.78 ms. Leave-one-family-out, MSLSAM-P reaches 46.3%:
better than any corpus-trained baseline, still worse than the zero-shot guards (27.8–30.5%). Across five further PCA seeds (v18b), the verdict held every time: the margin was 3.22 ± 0.13 points, with the same three
family wins. SPML stayed at 0.16–0.20% and DeepSet at 2.52–2.70%. The probe is also about five times more seed-stable than
the v17 pipeline, whose LOCO result ranges from 54.7% to 61.8% depending on the seed alone.

### 6.0c The fair comparison (v19): external-only training

Every trained method learned only from 5,916 external rows (hackaprompt, Gandalf, Alpaca, NotInject, LLMail-Inject in Enron inboxes,
SQuAD with injected instructions). None came from a test family, and a contamination guard dropped near-duplicates. MSLSAM-Δ uses the
**task-drift delta**: the hidden state at the end of the prompt minus the hidden state at the end of the user's task, from one forward pass.

| method | SPML ⚠ | DeepSet ⚠ | BIPIA | InjecAgent | pooled |
|---|---|---|---|---|---|
| MSLSAM-Δ (5 seeds) | 11.7 | 14.5 | **61.8** | 82.4 | 29.5 |
| Prompt Guard 2 | 1.6 | 20.2 | 82.0 | 53.4 | 28.4 |
| ProtectAI v2 | 1.0 | 27.5 | 83.1 | 63.4 | 29.2 |
| TF-IDF (same external data) | 29.7 | 39.1 | 75.9 | 81.1 | 46.0 |
| 8B judge (subsample) | 5.8 | 14.7 | 50.3 | 3.9 | 17.7 |

⚠ SPML and DeepSet are length-confounded; a length threshold alone gets 1.2% on SPML. Pre-registered outcome: MSLSAM **matches** the
guards pooled. On the length-clean families it is **complementary**: far better on documents/email (BIPIA), far worse on agent tool
output (InjecAgent), a format absent from its training data. It beats a text classifier trained on the same data in every seed.
The target (ASR < 5%) is not met under fair training.

### 6.0d External tool-output data (v20b), still no test-family data

Adding 1,500 external tool-output rows (glaive function-calling responses with LLMail/hackaprompt payloads and length-matched benign
twins) fixed the diagnosed failure: InjecAgent ASR fell from 82.4% to 55.2% (paired McNemar significant in 5/5 seeds). Pre-registered result:
**on the length-clean indirect families MSLSAM-Δ now beats Prompt Guard 2 in 5 of 5 seeds** (63.9% vs 78.2% ASR). Costs: BIPIA ASR +3.4
points, and false alarms on benign instructions inside tool output doubled (13.9% → 26.3%, the same as PG2's 26.8%). The target is not met.
The test families have now served several versions, so this should be confirmed once on unused data.

### 6.0e v21 — confirmation on fresh data, inference mode, complexity (run 2026-10-18)

**Result.**
- On 1,800 rows no version had seen, the frozen v20b MSLSAM-Δ **beats Prompt Guard 2 in 5 of 5 seeds** (pre-registered C1).
- FRESH pooled ASR at 10% FPR: 11.6% vs 38.1%. Sum-BIPIA 17.7 vs 67.3. Sum-TT, whose payloads were never used anywhere: 5.5 vs 8.8.
- The z-mean ensemble does **not** earn its cost overall (C2, 0/5). It helps on Sum-TT and hurts slightly on Sum-BIPIA.
  It reaches the target on Sum-TT alone: worst seed 3.33%, Wilson upper bound 6.0%, FPR 10.5%.
- **Costs stated with the win:**
  - MSLSAM flags harmless instructions inside articles 37–71% of the time (PG2: 10–14%).
  - Thresholds fixed on external data give 28–71% FPR on clean news, so per-context calibration is required.
- Full table: `versions/20_mslsam-v21/README.md`.


**Design.**
- v20b's MSLSAM-Δ is frozen: seeds 0–4 are refitted and must reproduce v20b's scores.
- It is evaluated once on 1,800 fresh rows:
  - BIPIA-style news summarization over XSum carriers
  - two payload sources: BIPIA text attacks, and Tensor Trust hijacks that no version had used
  - three length-matched tiers per carrier
- The verdicts use the v17b protocol per family.
- C1 compares MSLSAM-Δ with Prompt Guard 2 on FRESH pooled ASR.
- C2 tests a z-mean ensemble (μ/σ from external + tool benign rows only) against both of its components.

**Inference mode (what a deployment runs).**
1. The application assembles the chat prompt (system, task, untrusted data).
2. It runs **one** Llama prefill with `use_cache=True`. Forward hooks copy the hidden state at the last task token (anchor) and the last prompt
   token (end) at K = 8 layers.
3. The MSLSAM head (numpy) computes Δ = h_end − h_anchor, projects each layer with its PCA (k = 256), scales, and applies the LR, giving P(injection).
4. In the ensemble, Prompt Guard 2 scores the untrusted data. It can run in parallel, because it needs only the data text.
   The two logits are standardized and averaged.
5. The score is compared with the per-context threshold. If it is flagged, the request is blocked or sent back to the user.
   Otherwise the answer is decoded from the **same** KV cache, so the defence never adds a second prefill.

**Complexity.**
| component | time | memory |
|---|---|---|
| Llama prefill | O(L·(n·d² + n²·d)). It is shared: the application pays it anyway | — |
| hooks | O(K·d) | O(K·d) |
| MSLSAM head | O(K·d·k) ≈ 8.4 M multiply-adds, **independent of prompt length n** | O(K·d·k) ≈ 8.4 M parameters (33.6 MB fp32) |
| Prompt Guard 2 | O(⌈m/512⌉·L_g·(512·d_g² + 512²·d_g)) on m data tokens, from a separate model | ≈ 280 M parameters |
| 8B judge | a second full prefill | — |

Measured values come from three sources. v21c times the MSLSAM head on the frozen artifact, and v21b2 times Prompt Guard 2. The Llama prefill (326 ms) and the judge (378.5 ms) are v17's batch-1 measurements. v21's own Llama benchmark was abandoned because the machine was swapping 50 GB. So the hook overhead and decoding from the cache were not measured; they are given from theory.

### 6.0f v22 — reader-addressed instructions as benign (run 2026-10-18 — NEGATIVE)

**Definition change (owner's decision, 2026-10-18).** An *attack* is an instruction addressed to the AI that departs from the user's task.
A *benign* instruction belongs to the document and is addressed to its human reader (recipe and how-to steps, "please review the attached
draft"). v21's Dolly "hard negatives" are AI-addressed questions, so under this definition they are off-task requests. From v22 they are
reported as a flag rate ("AI-addressed off-task requests") and enter no verdict.

**Method.** Reader-addressed sentences are taken from instruction-following answers (Alpaca, Dolly), with AI-addressed cues filtered out.
1,200 are inserted into v20b's benign training carriers, in the same group as the carrier. 1,080 form a development tier inside the
hard-negative carriers of BIPIA, InjecAgent, Sum-BIPIA and Sum-TT. A0 (v20b, frozen) is compared with A1 (the same pipeline retrained with the
reader rows) under the v17b protocol.

**Pre-registered.**
- B0: A0's reader FPR ≤ 15% on every family.
- H1: A1 halves the mean reader FPR.
- H2: no detection loss (IND + 2 points at most, no McNemar loss).

H1 and H2 each need ≥ 4/5 seeds. v21's fresh families are used here as development data, so a pass needs a new fresh confirmation (v23).

**Result.**
- B0: the problem is real. A0's reader-tier FPR is 9.2 / 15.0 / 26.1 / 39.1% on BIPIA / InjecAgent / Sum-BIPIA / Sum-TT.
- H1 holds 5/5: the mean reader FPR falls from 22.3 to 10.4%.
- H2 fails 0/5: pooled indirect ASR rises from 48.1 to 60.1%, with McNemar losses on 3–4 families in every seed.
- v20b remains the detector.
- Post hoc, with the threshold set on clean + reader rows together: v20b 50.0%, A1 58.4%, Prompt Guard 2 68.5%, ProtectAI 68.1%.

**Interpretation.** The linear task-drift probe responds to instruction-like content in the untrusted data, whoever the instruction is
addressed to. Suppressing reader-addressed instructions suppresses detection of AI-addressed ones. Report this as a limitation:
26–39% reader FPR on summarization carriers, against 10–15% for the guards.

### 6.0g v23 — non-linear heads, a new fresh confirmation, GFLOPs (run 2026-10-19)

**Question.** v22 showed that a linear Δ-probe cannot separate reader-addressed from AI-addressed instructions without losing detection.
v23 asks two things:
- whether a head that can bend its boundary does better (MLR3, MLP, MLP on Δ + h_anchor), chosen on development data by v22's H1/H2
- whether v20b's win over Prompt Guard 2 replicates a second time, on families no version has used: IMDB reviews and unused BIPIA tables,
  each carrying Tensor Trust extraction attacks

**Cost.** FLOPs are counted with PyTorch's FLOP counter on the real model configurations, and checked against 2·P·n + 4·L·n²·d.

**Result.**
- **The screen:** no non-linear head kept detection while halving reader false alarms. H2 held in 0/5 seeds for every head.
- **The fresh confirmation (F1):** v20b lost to Prompt Guard 2 overall, 9.5% vs 5.6% pooled ASR. It won on IMDB reviews (2.6% vs 6.6%)
  and lost on tables (14.8% vs 4.8%), a carrier type absent from its training data.
- **Cost:** the head costs 0.017 GFLOPs per prompt, against 73 for Prompt Guard 2 and 5,265 for the protected model's own prefill (counted
  with PyTorch's FLOP counter, matching 2·P·n + 4·L·n²·d).
- **Post hoc:** v21's frozen z-mean ensemble of v20b and Prompt Guard 2 reached 0.9% pooled ASR, because the two detectors miss different
  attacks. It is reported as a hypothesis for pre-registered confirmation, since the same ensemble failed in v21.

### 6.0h v24 — confirming the frozen ensemble; tables in training (run 2026-10-19)

**Question.** v23 found that MSLSAM and Prompt Guard 2 miss different attacks, so v21's frozen z-mean ensemble reached 0.9% ASR post hoc.
v24 turns that into a pre-registered test:
- **Primary:** the ensemble, with every parameter fixed in v21, is scored once on two new families: news with human indirect injections
  (Gandalf-Summarization), and tables never used before, half in HTML, with Mosscap attacks. It must beat Prompt Guard 2 (E1) and reach
  ASR < 5% at FPR ≤ 10% on each family (E2).
- **Secondary:** MSLSAM retrained with table rows, which is adopted only if it improves tables on development data without losing elsewhere.

**Result.**
- **The ensemble (E1):** pre-registered and with every parameter fixed in v21, it beat Prompt Guard 2 in all five seeds (pooled ASR 8.75%
  vs 28.0%, McNemar wins on both families).
- **The target (E2):** reached on news (worst seed 0.99%) but not on tables (14.5%), where the remaining misses are very short
  password-seeking questions.
- **Tables in training:** fixed the development tables (14.8% → 0.6%) and improved the fresh tables (24.6% → 16.0%), but raised InjecAgent's
  ASR from 55% to 80%, so it was not adopted.
- **Cost:** the ensemble costs what Prompt Guard 2 costs, 1.8% of the protected model's prefill. The MSLSAM head adds 0.0003%.

### 6.0i v25 — single layer vs eight; which layers matter (run 2026-10-19)

**Result (pre-registered).** A single logistic-regression probe on the raw task-drift vector of **layer 12** (TaskTracker's probe type,
read from the application's own forward pass) beat the 8-layer MSLSAM head on all six fresh families. Pooled ASR at 10% FPR was 5.5% vs
12.3%, in 5/5 seeds. The layer was selected on development data only.
- **Ablation:** attack information peaks in layers 8–16, while false alarms on reader-addressed instructions fall with depth.
- **AUC vs ASR:** they rank detectors alike (ρ = −0.99), but AUC saturates near 1.
- **Cost:** the released, primed TaskTracker configuration needs two extra prefills (+138–147%).
- **Calibration:** about 100 clean documents per deployment context are needed to stabilise the threshold.
- **Consequence:** the "Why these 8 layers" rationale is superseded. The 8-layer head is reported as a negative result.

### 6.1 The headline

**Direct prompt injection, in-distribution — target met.**

| family | n | threshold | **ASR %** [Wilson 95%] | own-context FPR % |
|---|---|---|---|---|
| **SPML** | 2,499 | 0.006127 | **0.91 [0.61, 1.38]** | 9.91 |
| DeepSet | 263 | 0.011848 | 8.07 [5.28, 11.90] | 10.76 |
| BIPIA | 1,200 | 0.971377 | 31.26 [28.69, 33.93] | 10.52 |
| InjecAgent | 186 | 0.702137 | 47.06 [40.26, 54.47] | 10.10 |
| **pooled** | 4,148 | — | **12.21** | — |

Target was ASR < 5% with FPR < 10%. SPML meets it at n = 2,499. Indirect injection does not.

### 6.2 Numbers that must not be quoted

| number | why it is wrong |
|---|---|
| ASR 3.25% (v14) | pooled FPR of 9.99% concealed InjecAgent-Clean at **100%** and BIPIA-Clean at 47% |
| ASR 7.09% (v14) | thresholds fitted on the rows they were measured on, **and** an InjecAgent FPR estimated from **17 negative rows** (the real figure, on 479 rows, is 47.06% — not 19.35%) |
| v15 threshold-spread "collapse" | measured over 2 contexts because the chunk cache excluded the rest; restricted to those same 2, the previous version was already at 1.24× |

### 6.3 Two negative results worth reporting

**A payload is not a localised disturbance of the layer trajectory.** Version 15 cut each
document into proportional chunks, took hidden states at every chunk boundary from the same
forward pass, and used first differences to isolate each segment's contribution. On BIPIA
end-payloads — with the segment-0 confound removed — **no localiser beat a constant guess at
any depth** (best 43.0% exact / 79.5% within-1, against a constant predictor's 43.7% / 84.8%),
and `argmax ‖ΔZ‖`, which needs no training at all, sat at uniform chance. The segment where
the state moves most is not the segment holding the payload. **The signal is a whole-prompt
property, integrated across the context — not a step change when the payload is read.**

**The instruction/injection boundary is real but weak.** A classifier trained on exactly the
contrast — malicious instruction vs benign instruction, same position, same length, same
document — reaches AUC 0.8256 (BIPIA, carrier-grouped), 0.7984 (BIPIA, component-grouped),
0.8116 (InjecAgent). ASR 45–58% at a 10% budget. So the representation is *not* blind to
maliciousness, but the signal is far too weak to act on.

**Training on hard negatives is a trade, not a fix.** Including them lowers BIPIA-HardNeg FPR
79.26 → 57.40 and InjecAgent-HardNeg 22.03 → 8.08, but **InjecAgent detection collapses
76.88 → 29.57**, AUC falls 0.9820 → 0.9726 and ASR rises 5.30 → 8.27. It learned not to fire
on tool-output-shaped content at all, dropping false positives and true positives together.

### 6.4 The open problem

**Cross-context threshold calibration.** The cutoff each benign context needs for its own 10%
FPR spans **5.43 × 10³** — from 0.000179 (XSTest-Safe, short clean questions) to 0.973114
(BIPIA-Clean, retrieved documents). Retrieved documents and tool output sit intrinsically high;
short instructions sit at zero. Two attempts have failed: position-relative scoring (v15) and
hard-negative training (v16). Every version so far has assumed a single global operating point,
and that assumption now looks like the binding constraint.

---

## 7. Limitations

These are the boundaries of what the results support. They are not hedges.

1. **In-distribution only.** Leave-one-corpus-out: holding SPML out drops it from ASR 0.91% to
   **62.81%** (AUC 0.7790). BIPIA 82.73% (AUC 0.7031), InjecAgent 76.72%, DeepSet 58.66%. The
   detector recognises attack families it has seen, not injection in general.
2. **One model.** Llama-3-8B-Instruct only. Nothing establishes transfer to another model, and
   the features are tied to that model's representation geometry.
3. **One system prompt**, held fixed throughout. A deployed system varies it.
4. **No adaptive attacker.** Every attack is a static corpus entry. An attacker who knows the
   detector exists and optimises against the hidden-state signature has not been tested. For a
   security claim this is the most serious gap.
5. **English-dominant.** 12,897 of 13,245 rows; German is the only other language, at 348 rows.
6. **Indirect injection placed in the user turn.** BIPIA and InjecAgent deliver payloads via
   retrieved content or tool output; the monitor formats `system + user`, so carrier and payload
   both sit in the user turn. A standard simplification, but state it rather than let an
   examiner find it.
7. **InjecAgent is small and structurally narrow** — 17 tool-response templates and 62 attacker
   instructions. Its 186 injections are not 186 independent samples.
8. **No external benchmark.** Nothing here has been run against a held-out third-party
   benchmark. (Baseline detectors were run on this corpus in v17, §6.0.) See §9.

---

> **Updated by v17 (2026-09-28):** the 8B forward pass at batch size 1 measures **326.0 ms**
> (the earlier ~2,000 ms was batched throughput), so the marginal overhead is **0.99%**, not 0.16%.

## 8. Computational cost

Measured on the 24 GB M5 MacBook Pro, 2,000-row benchmark.

| component | per prompt |
|---|---|
| Llama-3-8B forward pass (bs=1, warm, MPS; v17) | **326.0 ms** |
| feature construction (24 features, 8 layers, 512 PCA dims, 4,148 prototypes) | **3.212 ms** |
| classifier inference | **0.005 ms** |
| **detector total** | **3.22 ms** |
| **marginal overhead on a pass already being run** | **+0.99%** |

| component | complexity | n = tokens, d = 4096, k = 512, L = 8, P = prototypes |
|---|---|---|
| forward pass | O(n²d + nd²) | dominates; one prefill |
| PCA projection | O(Ldk) | |
| prototype max-cosine | **O(LkP)** | **linear in the training set** |
| Mahalanobis | O(Lk²) | |
| HistGradientBoost | O(trees × depth) | effectively constant |

**The honest framing:** the guard costs one prefill of the prompt it is protecting, plus 3 ms.
If the protected model is running anyway, that is the *marginal* cost and it is very small. If
detection is compared as *total* compute, a dedicated 86M-parameter classifier is cheaper than
an 8B forward pass — so the argument is specifically about free-riding on a pass you already
need, and should be stated that way rather than as "3 ms".

**What will not scale:** prototype max-cosine is linear in the number of training prototypes
(currently 4,148). Cap the prototype set, or cluster it, before any production use.

---

## 9. What is missing before this is a general claim

1. ~~Baseline comparison on this corpus~~ — **done in v17 (§6.0)**. It went against MSLSAM on
   both pre-registered tests. A payload-grouped split for the indirect families is still missing.
2. **An external held-out benchmark** to rule out corpus-specific tuning.
3. **Adaptive-attacker evaluation** — the difference between "detects known attacks" and "is a
   defence".
4. **Per-context calibration**, the named open problem: a routing step that picks the threshold
   from content type instead of one global scalar.
5. **A second model**, to separate what is true of injections from what is true of Llama-3-8B.

---

## Appendix — version history

| # | version | mechanism | headline | status |
|---|---|---|---|---|
| 01 | GPT-2 pilot | attention entropy + trajectory | FPR 25.3%, ASR 75.1% | null result |
| 02 | Llama baseline | PCA → Mahalanobis + refusal direction | AUC 0.9495 | best classical |
| 03 | layer sweep | + dual threshold | AUC 0.9340; gray zone 14.5% | gray zone too wide |
| 04–08 | v2–v9 | MoE, injection direction, trajectory features | AUC 0.83–0.94 | score collapse at v9 |
| 09 | v10 | prototype max-cosine | injection AUC 0.9187 | ranking fixed |
| 10 | v11 | calibrated FPR budget + CV | AUC 0.9815, ASR 4.18% | threshold leak found later |
| 11 | v12 | grouped CV + nested thresholds | ASR 3.20% | generalisation limits found |
| 12 | v13 | cross-source FPR re-measure | no config meets both targets | overturned v12 |
| 13 | v14 | 4-corpus set + **classifier sweep** | **AUC 0.9867**, v02 confound solved | the turning point |
| 14 | v15 | position-relative trajectory | premise **falsified** | two findings kept |
| 15 | v16 | hard-negative training | **SPML ASR 0.91%**; pooled 12.21% | scoped positive |
