# 19 — MSLSAM v20 (turning complementary errors into one detector)

**Status:** ✅ v20 COMPLETE 2026-09-29 — v20a ensemble: E1-LC does NOT hold (0/5) · **v20b tool data: T1 HOLDS (InjecAgent −27 pts, 5/5), T2 FAILS (BIPIA +3.4, 1/5), T3 BEATS Prompt Guard 2 on the length-clean families (5/5)** · target not met

## Why
v19, the fair comparison with external-only training, found MSLSAM-Δ and the zero-shot guards **complementary**:

| | BIPIA (documents/email) | InjecAgent (tool output) |
|---|---|---|
| MSLSAM-Δ | **61.8%** ASR | 82.4% |
| Prompt Guard 2 | 82.0% | **53.4%** |

- **v20a** asks whether an ensemble turns that into a detector better than either component.
- **v20b** attacks the other cause: MSLSAM-Δ never saw tool-output text in training.

## v20a design — nothing fitted on test data
- **Components:** MSLSAM-Δ (v19c's test scores, 5 PCA seeds) and **Prompt Guard 2** on the untrusted data. PG2 was v19's best
  full-coverage guard; it is fixed now and not chosen after the fact.
- **A common scale:** each detector's score → `z = (logit(score) − μ) / σ`, where μ and σ come from that detector's scores on the **external
  benign rows**. For MSLSAM-Δ these are **out-of-fold** external scores (5-fold grouped CV, C = 0.01 as v19c chose), so the
  reference is not in-sample. The guards are scored on the 5,916 external rows for this (v19b2's code).
- **PRIMARY: z-max** = max(z_Δ, z_PG2), an OR-ensemble that flags when either detector is unusually confident. No free parameters.
- Secondary (reported, never decisive): z-mean; stacking LR (C = 1, balanced) on [z_Δ, z_PG2], fitted on external rows with OOF z_Δ;
  a three-way z-max adding ProtectAI; and z-max(Δ, 8B judge) on the judge's subsample.
- Thresholds: v17b's nested per-family 10% FPR protocol, unchanged.

## v20a pre-registered reading (fixed 2026-09-29, before any v20 number existed)
- **E1-LC (GOVERNING, BIPIA + InjecAgent, the length-clean families):** the z-max ensemble's pooled ASR must be **≥ 2 points below the
  better of its two components**, with **no McNemar loss against either component** on either family and **≥ 1 family win**
  against at least one of them. It must hold in **≥ 4 of 5 seeds**; 2–3 counts as seed-dependent.
- **E1-ALL:** the same rule over all four families, reported. SPML and DeepSet carry the v19 length flag.
- **Target (ASR < 5%):** counted per family only if the **worst** seed is < 5%.
- **Cost, stated with the result:** MSLSAM-Δ 0.81 ms on the shared forward pass + PG2 32.4 ms as a separate model ≈ 33 ms, against 378 ms for the 8B judge.
- Cannot be changed after the run: the component guard, the z-reference (external benign), the combiner (max), C, or the seeds.

## v20b (planned — superseded by the v20b section below)
Add externally sourced **tool-output** injections, e.g. an agent benchmark other than InjecAgent, to the external training set.
Re-extract only the new rows (~1–2 h GPU) and retrain MSLSAM-Δ. The question: does the InjecAgent gap close without touching InjecAgent?
Data sourcing and pre-registration come after v20a's result.

## v20a dry run
The dry run reused v19c's synthetic stand-in states and the real v19 CSVs, with guard scores stubbed. All cells passed: OOF fits,
z-standardisation, all ensembles, the verdict logic and the saves. The dry-run numbers are meaningless.


## v20a — results (run 2026-09-29, ~15 min)

On the external rows the guards score AUC 0.949 (PG2) and 0.950 (ProtectAI). MSLSAM-Δ's out-of-fold AUC is 0.997 in every seed.

ASR % at own 10% FPR, mean of 5 seeds:

| method | SPML ⚠ | DeepSet ⚠ | BIPIA | InjecAgent | pooled | indirect |
|---|---|---|---|---|---|---|
| MSLSAM-Δ | 11.67 | 14.51 | 61.79 | 82.36 | 29.52 | 64.55 |
| **z-max ensemble [PRIMARY]** | 7.77 | 14.38 | 61.39 | **82.36** | 27.05 | 64.21 |
| z-mean (secondary) | 5.81 | 10.53 | **53.54** | **74.35** | **22.99** | **56.33** |
| stacking (secondary) | 8.14 | 12.99 | 59.28 | 80.20 | 26.47 | 62.09 |
| z-max + ProtectAI (secondary) | 0.96 | 14.42 | 63.66 | 82.36 | 23.60 | 66.17 |
| Prompt Guard 2 | 1.59 | 20.23 | 82.03 | 53.37 | 28.36 | 78.18 |

**Pre-registered verdict: E1-LC does not hold (0/5 seeds); E1-ALL does not hold (0/5).** The ensemble is only ~0.3 points better than
MSLSAM-Δ on the indirect families, and PG2 beats it significantly on InjecAgent in every seed. Target: not met.

### Why the OR-ensemble failed (diagnosis, seed 0)
Median z-scores on the external-benign scale:

| corpus | z_Δ | z_PG2 | rows where Δ dominates the max |
|---|---|---|---|
| InjecAgent (injections) | 3.96 | 0.09 | 99.5% |
| **InjecAgent-Clean (benign)** | **3.60** | -0.14 | **100%** |
| BIPIA | 3.73 | 0.29 | 94.5% |
| BIPIA-Clean | 1.04 | 0.11 | 70.4% |

MSLSAM-Δ sits **~3.6 SD above its external-benign reference on every InjecAgent row, benign or not**. Tool output is out of
distribution for it. PG2's InjecAgent signal lives in tiny absolute scores (z ≈ 0), so `max` picks Δ on 100% of rows, and the
ensemble *is* MSLSAM-Δ there. The general lesson: **calibrating on external benign data does not transfer across a domain shift,
and an OR-ensemble is hostage to whichever component shifts.** This points back at v20b: give MSLSAM-Δ tool-output training data.

### Not claimable, but recorded
z-mean, which was pre-registered as *secondary*, beat both components descriptively: indirect 56.3 vs 64.6 (Δ) and 78.2 (PG2), pooled
23.0 vs 29.5 / 28.4. It was not the primary rule, and the test set has now informed that observation. Promoting it would be
choosing a rule after seeing the results. It can only become a claim if it is re-tested, pre-registered, on data it has not seen
(e.g. after v20b, or on a new held-out set).

## v20b — external TOOL-OUTPUT training data (built 2026-09-29, pre-registered before any v20b number)

**Why:** v19 and v20a both trace the InjecAgent failure to one cause. MSLSAM-Δ has never seen tool output, so every tool-output row,
benign or not, sits ~3.6 SD off its external-benign scale (v20a diagnosis).

**Data (v20b1):** 750 **paired** rows from `glaiveai/glaive-function-calling-v2` (Apache-2.0), with real function-calling dialogues.
- task = the user's request
- data = `Tool result (<function>): <JSON>`, written as JSON with double quotes. InjecAgent's test rows use Python-repr quotes, so this is
  not a copy of the test format.
- **injected:** the JSON's longest string field is replaced by an external payload. Half are LLMail-Inject and half hackaprompt, **all
  disjoint from v19's payloads**.
- **benign twin:** the same field is replaced by a length-matched *declarative* SQuAD filler. This is InjecAgent's own tier-1 construction.
  No instruction-shaped negatives.
- v19a's contamination guard and ≤ 1,024-token check are applied unchanged.

**Extraction (v20b2):** v19b's code, unchanged; 1,500 rows, ~30–45 min.

**Model (v20b3):** v19c's MSLSAM-Δ pipeline, unchanged: C from v19c's grid by inner CV, PCA seeds 0–4. It is trained on the 5,916 v19
external rows **+ 1,500** tool rows. Still no test-family data.

### Pre-registered reading (fixed 2026-09-29, before any v20b number existed)
Comparisons are **paired, seed by seed**, with v19c's MSLSAM-Δ (the same seed, the same pipeline, without the tool rows).
- **T1: does the InjecAgent gap close?** The mean InjecAgent ASR over 5 seeds must drop by **≥ 10 points** (v19c: 82.36 → ≤ 72.36),
  **and** McNemar must favour v20b on InjecAgent in **≥ 4 of 5** seeds.
- **T2: no collateral damage.** Per seed, BIPIA ASR must be ≤ v19c + 2 points, with no McNemar loss on BIPIA, in ≥ 4 of 5 seeds.
- **T3: the fair headline, re-run.** Length-clean F1 against Prompt Guard 2 (BIPIA + InjecAgent), using v19's rule: BEATS / MATCHES /
  LOSES, ≥ 4 of 5 seeds.
- Reported, never decisive: InjecAgent-HardNeg FPR (does tool training raise false alarms on benign tool-output instructions?),
  SPML and DeepSet (length-flagged), and the target (worst seed < 5%).
- **Caveat stated in advance:** InjecAgent stays a *transfer* test. Glaive's tools and InjecAgent's tools are different
  APIs, and none of InjecAgent's payloads, templates or tools are in training. What is shared is the *genre*: a JSON tool response
  following a user request. That is the point: a deployed agent's detector would be trained on tool-output traffic of that genre.

### Dry runs
v20b1 ran on a stand-in glaive file in the real format, with the real v19 CSVs; the length gap after multi-sentence filler matching was 7%.
v20b2 ran on the v19b mock model: anchor after `…?\n\n`, next token `Tool result`. v20b3 ran on synthetic states. All passed; their numbers are meaningless.

### v20b1 run 1 — crashed at the pairing cell (2026-09-29): the filler pool ran dry
`IndexError: pop from empty list` in `take_sentence`. Two bugs of mine. (1) `json.dumps` escapes quotes and backslashes, so a payload
containing `"` never matched its own escaped form inside the JSON text, and the row was rejected. (2) The benign filler was drawn **before**
that check, so every rejected candidate consumed filler sentences until the pool emptied. **Fix:** the check now compares the *escaped*
payload (which is also what is stored as `piece`); a payload that still cannot be used is dropped; the filler is drawn only after the
injected row is valid; the SQuAD sample grew from 6k to 20k paragraphs; and an assertion fires before the pool can empty. The dry run
was repeated with payloads containing quotes and backslashes: 750 pairs, length gap 6.9%. No v20b number existed yet.


## v20b — results (run 2026-09-29)
**Build:** 21,134 usable glaive dialogues. Payload pools after removing v19's payloads: LLMail 19,667, hackaprompt 15,295; 6 + 102 test near-duplicates
dropped. **750 pairs; injected and benign median length both 454 chars (gap 0.0%).** Tokens p50 151, max 398.
**Extraction:** anchor after the user request (`…Cooking Class".\n\n`), next token `Tool result (`; 2.3–3.1 rows/s; 6/6 shards.
**Model:** C = 0.01 in every seed (as in v19c).

| seed | InjecAgent v19c → v20b | McNemar (v20b only / v19c only) | BIPIA v19c → v20b | indirect v20b | InjecAgent-HardNeg FPR v19c → v20b |
|---|---|---|---|---|---|
| 0 | 81.4 → **48.9** | 68 / 8 | 58.1 → 64.2 | 62.2 | 14.5 → 31.2 |
| 1 | 80.7 → **63.0** | 49 / 15 | 63.9 → 65.7 | 65.3 | 13.4 → 23.2 |
| 2 | 84.0 → **54.5** | 66 / 11 | 63.3 → 64.0 | 62.7 | 12.7 → 25.2 |
| 3 | 85.1 → **54.7** | 66 / 9 | 61.9 → 65.5 | 64.1 | 13.9 → 27.3 |
| 4 | 80.7 → **55.0** | 58 / 11 | 61.8 → 66.7 | 65.2 | 15.0 → 24.8 |
| **mean** | **82.4 → 55.2** | all p < 0.001 | **61.8 → 65.2** | **63.9** | **13.9 → 26.3** |

Guards for reference: Prompt Guard 2 has BIPIA 82.0, InjecAgent 53.4, indirect 78.2, InjecAgent-HardNeg FPR 26.8. ProtectAI has 83.1 / 63.4 / 80.5 / 8.1.
SPML and DeepSet are essentially unchanged (SPML ~10.8, DeepSet ~13.8). Pooled over all four families: 28.7 (PG2 28.4).

### The pre-registered verdict, as fired
- **T1 — the InjecAgent gap closes: HOLDS.** The mean drops by **27.2 points** (82.4 → 55.2), and McNemar favours v20b in **5 of 5** seeds.
- **T2 — no collateral damage on BIPIA: DOES NOT HOLD (1/5).** BIPIA gets worse by +0.8 to +6.2 points (mean +3.4). The tool data costs
  some document/email sensitivity.
- **T3 — the fair headline, length-clean F1 against Prompt Guard 2: BEATS in 5 of 5 seeds.** Indirect ASR is 63.9 vs 78.2 (12.9–16.0 points
  better). McNemar gives MSLSAM BIPIA in every seed, and **InjecAgent is no longer a significant loss** (55.2 vs 53.4).
- **Target:** not met on any family.

### What this means
1. **The first pre-registered result in which MSLSAM, trained only on external data, beats the best purpose-trained guard on the
   length-clean (indirect) families.** It costs 0.81 ms on top of the forward pass the model runs anyway; PG2 is a separate 32 ms model.
2. **The fix worked for the diagnosed reason.** Benign tool-output examples taught MSLSAM that "looks like tool output" is not suspicious,
   and InjecAgent ASR fell by 27 points. None of InjecAgent's payloads, templates or tools were in training. The shared part is the genre
   and the application's wrapper string (`Tool result (<fn>):`), which appears in benign and injected training rows alike and so
   cannot act as a label shortcut.
3. **The costs must travel with the claim:**
   - BIPIA got worse by 3.4 points (T2 failed).
   - **False alarms on benign instructions inside tool output roughly doubled (13.9% → 26.3%)**, to PG2's level (26.8%).
     MSLSAM now flags tool output that *contains an instruction* more readily, whether the instruction is benign or malicious. This is the v15/v16
     instruction-vs-injection boundary again.
4. **The 8B judge is still the strongest indirect detector** (InjecAgent 3.9%, BIPIA 50.3%), at 378 ms per call.

### A caution about the test set
This test set has now been used for v17 → v20b. Every verdict was pre-registered, but across versions the *design choices*
(the tool data, the Δ feature) were informed by earlier results on the same test families. **A claim for the thesis should be confirmed once on
data never used in any version.** BIPIA's unused `qa` and `abstract` tasks, for instance, or a second agent benchmark. That is the natural v21.
