# 21 — MSLSAM v22 (stop flagging instructions that belong to the document)

**Status:** ✅ RUN 2026-10-18 — **NEGATIVE: H1 holds (5/5), H2 fails (0/5); v20b stays the detector.** Built 2026-10-18. All four notebooks passed a dry run in the cloud workspace (v22a on mock Alpaca; v22b on a tiny random 32-layer Llama, including resume; v22b2 on keyword stand-in guards, including resume; v22c on synthetic hidden states, exercising both the PASS and the FAIL branch and the resume path). **The pre-registration below was fixed before any v22 number existed.**

## Why
v21 confirmed that MSLSAM-Δ beats Prompt Guard 2 on unseen indirect injection (FRESH ASR 11.6 vs 38.1, 5/5 seeds). It also showed the
main weakness: MSLSAM flags the "harmless instruction" tier 37–71% of the time, where PG2 flags it 10–14%.

**The definition (decided 2026-10-18).** That tier is built from Dolly questions addressed to an AI ("Who designed Augusta National?"),
pasted into a document the user asked to have summarised. Those pull the model off its task, which is BIPIA's own definition of an
injection. The benign instructions that really occur in documents are **addressed to the document's human reader**: recipe and how-to
steps, manual steps, an email's "please review the attached draft". From v22 on:

- **attack:** an instruction aimed at the AI that departs from the user's task (BIPIA, InjecAgent and Tensor Trust payloads, and the old Dolly tier)
- **benign:** an instruction that belongs to the document and is addressed to its reader

The old Dolly tier is kept, renamed **"AI-addressed off-task requests"**, and reported as a detection rate. It enters no verdict.

## Design

### New data (v22a, CPU)
Reader-addressed instructions are imperative sentences taken from instruction-following *answers*, such as "Preheat the oven to 180 °C." or
"Make sure to disable power to the bulb before removing it." Sentences that address an AI are filtered out: "you are", "assistant",
"ignore", "instruction", "respond", "answer", "output", "say", "summarise", "translate", "password", and similar words. Questions and quote
characters are dropped too.

- **Training, 1,200 rows, label benign:** Alpaca answers (rows of split A, 2/3 — Amendment 1). Each row reuses a v19/v20 *benign* training carrier, a SQuAD
  paragraph, an Enron inbox or a glaive tool output, with the same task, position and group. The declarative filler is replaced by a
  length-matched reader instruction, 400 rows per carrier type. Every reader row therefore sits beside the injected and the declarative
  versions of the same carrier.
- **Development test, 1,080 rows, label benign:** Dolly answers plus Alpaca rows of split B (1/3). Neither shares a row with training, and the two are
  checked for 8-gram overlap. Each row reuses an existing hard-negative row (the same carrier, position and task) with the Dolly question
  replaced by a length-matched reader instruction:
  - BIPIA-Reader: 240
  - InjecAgent-Reader: 240
  - Sum-BIPIA-Reader: 300
  - Sum-TT-Reader: 300
- **Contamination:** training text is checked against all test and dev text (the v19a guard: 50% of word 8-grams).
- **Token check:** every row must fit 1,024 tokens.

### Extraction (v22b, GPU, ~35–45 min) · guards (v22b2, no Llama, ~3 min)
- v22b uses v19b's extraction code unchanged, with the same pre-flight gate (MIN_RATE, FORCE = False).
- v22b2 scores the 1,080 development rows with Prompt Guard 2 and ProtectAI. It reuses v21b2's fixed batching (padded to a multiple of 64, empty_cache).

### Arms (v22c, CPU), seeds 0–4
- **A0 = v20b, frozen.** The `artifacts/` numpy head, unchanged.
- **A1 = v20b's data + the 1,200 reader rows.** Same pipeline: PCA(256) per layer with random_state = seed, then StandardScaler, then
  LR (balanced), with C from the same grid by the same inner grouped 3-fold CV.

### Evaluation: the v17b protocol, unchanged
- Each family is scored at its own 10% FPR budget, set on its **clean** tier.
- The reader tier and the Dolly tier are reported as extra FPR / flag rates at that same threshold.
- Families: BIPIA, InjecAgent, Sum-BIPIA, Sum-TT. SPML and DeepSet are reported descriptively.
- **v21's fresh families become development data here.** They are no longer "unseen", and that is disclosed.

## Pre-registered reading (fixed 2026-10-18, before any v22 number)
**RF(m)** = the mean, over the 4 indirect families, of m's reader-tier FPR.
**IND(m)** = the pooled ASR over the 4 indirect families. Each rule is checked per seed and must hold in **≥ 4 of 5 seeds**.

- **B0 (baseline check):** if A0's reader-tier FPR (mean of seeds) is ≤ 15% on every family, the v21 over-flagging was specific to
  AI-addressed requests. The thesis then reports that, and no retraining is needed. A1 is still reported.
- **H1 — fewer false alarms:** RF(A1) ≤ 0.5 × RF(A0).
- **H2 — no loss of detection:** IND(A1) ≤ IND(A0) + 2 points, and McNemar(A1 vs A0) shows no loss on any of the 4 families.
- **Decision:**
  - If H1 and H2 both hold, A1 becomes **MSLSAM v22**. Its artifacts are saved as the frozen candidate for a later fresh confirmation (v23,
    not in this version's scope). v22c asserts that the frozen `artifacts/mslsam_v22_s*.npz` reproduce A1's scores through the numpy head
    (< 1e-4), and that A0 reproduces v20b's and v21's scores before anything is compared.
  - Otherwise v20b stays the detector, and v22 is reported as a negative result.
- **Descriptive only:**
  - the Dolly tier's detection rate
  - PG2 and ProtectAI on the reader tier
  - SPML / DeepSet
  - per-family ASR
  - the target

## Amendment 1 (2026-10-18, before any v22 number existed)
v22a run 1 stopped while building the training rows with `reader-sentence pool exhausted`. With a 1/2 split, Alpaca A gave **2,835**
training sentences, and the length-matched draw needed about **2,900**. Nothing was written and no v22 number exists, so two build
details were changed. The definition, filters, carriers, row counts, arms and rules are unchanged.
- **Alpaca split:** 2/3 → training, 1/3 → development (was 1/2 : 1/2). Dolly stays development-only.
- **Mail rows:** the email keeps its start and ends with a **1–2 sentence** reader instruction (target length = email length / 4, clipped to
  40–160 characters; was / 2, clipped to 60–400). A real email ends with "please review the attached draft", not a paragraph of steps.
  The total length of the row is still matched, because the email is cut by the length of the instruction.
- v22a now prints how much of each pool it used.

**Cannot be changed after the run:** the filters, the sources, the row counts, the C grid, the seeds, the rules.

## How to run (in this order)
| step | notebook | where | time | note |
|---|---|---|---|---|
| 1 | `v22a_build.ipynb` | CPU, any kernel | ~2–5 min | downloads Alpaca once (~25 MB); stops with "pool exhausted" or a contamination assert if anything is off |
| 2 | **restart the Mac**, check swap | | | `sysctl vm.swapusage`: used should be < 2 GB |
| 3 | `v22b_extract.ipynb` | MPS, fresh kernel, nothing else open | ~35–45 min for 2,280 rows | stops at the smoke gate if < 0.20 rows/s; resumable per shard of 250 |
| 4 | Kernel → Restart | | | never load the guards next to Llama |
| 5 | `v22b2_guards.ipynb` | MPS | ~3 min | resumable per guard |
| 6 | `v22c_analysis.ipynb` | CPU | ~15–30 min (5 seeds × PCA + inner CV) | resumable per seed |

## Run log
- **v22a run 1 (2026-10-18):** stopped in cell 3, `reader-sentence pool exhausted`. Training pool 2,835 sentences, demand ~2,900. Nothing
  written. Fixed by Amendment 1.
- **v22a run 2 (2026-10-18 23:27): OK.** Training pool 3,764 sentences, 2,105 used (56%); development pool 2,969 (108 dropped as near
  training), 1,455 used (49%). 0 training sentences near test/dev text. Rows: Doc/Mail/Tool-Reader 400 each; BIPIA/InjecAgent-Reader 240,
  Sum-BIPIA/Sum-TT-Reader 300. Replaced and new pieces have the same median length per family (66/108/62/151 chars). Tokens: training
  median 223, max 749; development median 364, max 994 (limit 1,024). No duplicate texts; 0 pieces shared between training and development.
  - Audit (Claude, read-only): about 0.5% of training sentences (11/2,105) read like tasks for an AI ("Create a program for the robot to
    execute.", "Create a list of potential passwords ..."; the filter matches "password" but not "passwords"); 2/1,455 in development.
    Multi-sentence pieces join sentences from different answers, so some read as unrelated tips. Both are disclosed label/realism noise,
    not changed after the build.
- **v22b / v22b2 / v22c (2026-10-18): OK.** See the result below.

## RESULT (run 2026-10-18) — B0: problem real · H1 HOLDS (5/5) · H2 does not hold (0/5) → NEGATIVE: v20b stays the detector

**Run log.**
- v22b: the swap check passed (1.9 GB used). The speed check passed, 0.25 rows/s warm. 2,280 rows were extracted in about 30 min (0.7–2.4 rows/s per shard).
- v22b2: Prompt Guard 2 took 47 s and ProtectAI 38 s. Device memory stayed at 1.88 GB.
- v22c: A0 reproduced v20b and v21 (< 1e-4). The inner CV picked C = 0.01 in every seed, the same as v20b. Clean-tier FPR was about 10.7% by construction.

**Pre-registered verdict** (each figure is the mean of 5 seeds; RF = mean reader-tier FPR; IND = pooled indirect ASR):

| | A0 (v20b) | A1 (+ reader rows) | Prompt Guard 2 | ProtectAI |
|---|---|---|---|---|
| reader FPR: BIPIA / InjecAgent / Sum-BIPIA / Sum-TT | 9.2 / 15.0 / 26.1 / 39.1 | 3.5 / 10.8 / 13.6 / 13.4 | 16.2 / 21.9 / 11.8 / 10.2 | 9.1 / 8.3 / 11.8 / 15.3 |
| **RF** | **22.3** | **10.4** | 15.0 | 11.1 |
| ASR: BIPIA / InjecAgent / Sum-BIPIA / Sum-TT | 65.2 / 55.2 / 17.7 / 5.5 | 78.1 / 67.9 / 35.4 / 7.8 | 82.0 / 53.4 / 67.3 / 8.8 | 83.1 / 63.4 / 75.1 / 4.0 |
| **IND pooled ASR** | **48.1** | **60.1** | 66.1 | 68.1 |
| SPML / DeepSet ASR | 10.8 / 13.8 | 25.5 / 15.9 | 1.6 / 20.2 | 1.0 / 27.5 |
| Dolly tier flagged (AI-addressed, descriptive) | 8 / 26 / 37 / 71 | 3 / 20 / 21 / 50 | 11 / 27 / 10 / 14 | 7 / 8 / 10 / 12 |

- **B0:** A0's reader-tier FPR is 9.2 / 15.0 / 26.1 / 39.1, so the problem is real. MSLSAM also flags instructions addressed to the reader, mostly in the summarization carriers (Sum-TT 39%). On the old BIPIA / InjecAgent carriers it is near budget.
- **H1 HOLDS, 5/5:** RF falls from 22.3 to 10.4, a 53% cut. It is now at the guards' level (PG2 15.0, ProtectAI 11.1).
- **H2 does not hold, 0/5:** IND rises from 48.1 to 60.1, which is 12 points worse. McNemar shows a loss on BIPIA, InjecAgent and Sum-BIPIA in every seed, and on Sum-TT in 3/5 seeds. The worst case is BIPIA seed 0: 171 attacks caught only by A0, 17 only by A1. SPML ASR also doubles, from 11 to 26.
- **Decision (pre-registered):** v20b stays the detector, and v22 is reported as a negative result. No `artifacts/mslsam_v22_*.npz` was written; the A1 candidates stay in `artifacts/candidates/`.

**Post hoc** (not pre-registered; `results/posthoc_pooled_benign.csv`, `code/posthoc_pooled_benign.py`). If the threshold is set on the v22 definition's full benign set (clean + reader rows together, 10% FPR):

| | A0 | A1 | PG2 | ProtectAI |
|---|---|---|---|---|
| IND pooled ASR (seed mean) | **50.0** (48.9–51.1) | 58.4 | 68.5 | 68.1 |
| Sum-TT ASR | 12.2 | **8.4** | 8.9 | 4.2 |

Under the new definition the frozen v20b is still the best of the four on the indirect families, 18 points ahead of the guards. A1 wins only on Sum-TT.

**What this means.**
1. The linear Δ-probe does not separate "an instruction addressed to the reader" from "an instruction addressed to the AI". When it is taught that reader imperatives are benign, it gives up part of the signal it uses to catch injections. That signal is the presence of instruction-like content in the data, and it is shared by both kinds of instruction. A1 also flags fewer of the AI-addressed Dolly requests (71 → 50% on Sum-TT), which under the v22 definition are attacks.
2. This is a trade-off along one line, not a fix. The figure `figures/v22_tradeoff.png` shows A1 moving from bottom-right to top-left, and both arms stay below both guards.
3. The thesis claim stays v20b's: it beats PG2 on unseen indirect injection (v21 C1). The v22 limitation, written as a measured number: on summarization carriers MSLSAM flags reader-addressed instructions at 26–39% at the clean-calibrated threshold, against 10–15% for the guards.
4. Separating "who the instruction is addressed to" probably needs a non-linear head, or features from the task–data relation rather than Δ alone. That is future work, not v22.

**Caveats.**
- The Sum-* families were already used in v21, so they are development data here.
- About 0.5% of the training reader sentences read like tasks for an AI.
- The reader tier is built from Alpaca/Dolly answers. These are realistic instructions but synthetic documents.
