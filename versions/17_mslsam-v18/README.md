# 17 — MSLSAM v18 (MSLSAM-P: direct probe, leak-free split)

**Status:** ✅ v18 run 1 + v18b COMPLETE — **H2 STABLE (5/5 seeds)**, H1 not (0/5), H3 holds (5/5); v19 recommended · **Source:** `v18_probe.ipynb` (CPU only; Llama not loaded; ~45–75 min; resumable)



**CPU only. Llama is not loaded.** Everything reads cached states (`representation_cache/joined_v16`,
13,245 rows × 8 layers × 4,096) and v17's cached MiniLM embeddings. Expect roughly **45–75 min**;
every score file is saved as it is produced, so an interrupted run resumes where it stopped.

## Why

v17 established two things. First, a TF-IDF classifier trained on the same folds beat MSLSAM
in-distribution (6.07% vs 14.42% pooled ASR). Second, v17's folds hold out *carriers* but not
*payloads*: 96–98% of BIPIA and InjecAgent test injections had their exact attack text in training.
v18 asks the question v17 could not answer:

> **Is it Llama's hidden states that lose to text, or MSLSAM's 24-number summary of them?**

MSLSAM v17 compresses 8 layers × 4,096 dimensions into 24 scalars (prototype max-cosine and
Mahalanobis distance). **MSLSAM-P** skips that compression and trains a probe directly on the states.

## MSLSAM-P, fixed before the run

| step | setting |
|---|---|
| input | last-token hidden state, layers 4, 8, 12, 16, 20, 24, 28, 31 (same cache as v14–v17) |
| reduce | per layer, `PCA(256, random_state=42)` fitted on the **training fold only** → 8 × 256 = 2,048 features |
| scale | `StandardScaler`, training fold only |
| **primary head** | `LogisticRegression(class_weight="balanced")`, C ∈ {0.001, 0.01, 0.1, 1} chosen by **inner** 3-fold grouped CV on the training fold (ROC-AUC) — the test fold never touches C |
| secondary head | `HistGradientBoostingClassifier(max_iter=300)`, same features — **reported, but cannot establish a claim on its own** |

## Four protocols, applied identically to every trained method

| protocol | what is held out of training | role |
|---|---|---|
| **P1** | carrier (v17's exact folds, loaded from disk) | continuity with v17 only |
| **P2** | **payload string** for BIPIA / InjecAgent injections; carrier/payload elsewhere | **PRIMARY** |
| **P2p** | P2 **plus** every row sharing a carrier with the test fold's indirect injections | robustness: nothing about the test attack *or* its document was seen |
| **LOCO** | the whole family | transfer (as v17) |

The trained methods are MSLSAM-P (LR, HGB), MSLSAM v17 (reference), TF-IDF + LR, MiniLM + LR and MiniLM + HGB. Each one is
refitted on every split. The zero-shot methods (ProtectAI, Prompt Guard 2, the 8B judge and the rules)
do not train, so their v17 scores are reused unchanged as reference rows.
Threshold protocol: **v17b's own code, copied verbatim** — nested per-family 10% FPR, 400 resamples, tie-aware.

## Pre-registered reading (written before any v18 number exists)

Pooled ASR = injection-weighted ASR over SPML, DeepSet, BIPIA and InjecAgent, each at its own
10% FPR. "Beats" means lower pooled ASR. McNemar is paired on identical injections, with p < 0.05.

- **H1 — bottleneck.** MSLSAM-P (LR) on P2 is ≥ 2 pooled points below MSLSAM v17 on P2
  → *the 24-feature summary was throwing signal away.*
- **H2 — hidden states beat text, in-corpus.** On P2, MSLSAM-P (LR) is ≥ 2 pooled points below
  TF-IDF + LR, **and** McNemar favours MSLSAM-P on ≥ 2 families, **and** favours TF-IDF on none.
  **H2 also needs robustness:** on P2p, MSLSAM-P (LR) must not have a higher pooled ASR than TF-IDF.
  If TF-IDF is within 2 points or better → *no representational advantage in-corpus. Say so plainly.*
- **H3 — transfer retained.** MSLSAM-P (LR) LOCO pooled ASR is no more than 2 points above
  MSLSAM v17's LOCO → *the probe keeps the cross-family advantage v17 found.*
- **v19 (task-drift Δ, ~7 h of GPU) is recommended only if H2 holds, robustness included.**
  Otherwise, write the thesis up as the evaluation study (v17 + v18).
- **Target (ASR < 5%, FPR < 10%)** is reported per family under P2 for every method. It is
  descriptive and does not change H1–H3.

Things that **cannot** rescue a failed hypothesis afterwards: the HGB head, P1 numbers, a different
C grid, a different layer subset, or dropping a family. Anything like that goes in a new version,
with its own pre-registration.

## Integrity checks built in
- P1 folds are loaded from `results/v17/scores/_folds_indist.npy`, not recomputed.
- **TF-IDF on P1 must reproduce v17c's saved scores.** If it does not, the notebook stops (row order or
  folds would be wrong).
- Leak table: P2 and P2p must show **0.0% payload seen** (asserted), and P2p must show **0.0% carrier seen** (asserted).
- MSLSAM v17 on P1 is printed next to v17's 14.42%. Expect a difference of up to ~0.6 points,
  because PCA was unseeded in v17 run 2 and is seeded now.

## How to run

1. Any kernel. Llama is **not** loaded, so no fresh-kernel requirement. Memory: ~2 GB of cached states.
2. Run all cells top to bottom. Every score is saved to `results/v18/scores/<method>__<protocol>.npy`
   as it is produced. If the run stops, re-running skips everything already on disk.
3. To save ~15–25 min, set `RUN_HGB_PROBE = False` in cell 1. It only drops the secondary head.
4. Report back cell 1 (leak table), cell 6 (the reproduce lines), cell 7 (tables) and cell 8 (VERDICT).

## Dry run (2026-09-28), and one disclosure

The notebook was dry-run end-to-end in the cloud workspace. It used the **real** `rows_v16.csv`, the real
corpus text, v17's real fold file and the real TF-IDF v17 scores. The hidden states, MiniLM embeddings and
zero-shot scores were **synthetic**, so the dry run's MSLSAM and MiniLM numbers mean nothing. All 15 corpora
were present, which is the lesson from v17c.

- Leak assertions pass on the real metadata: P2 and P2p show 0.0% payload seen, and P2p shows 0.0% carrier seen.
  InjecAgent has 62 payload strings. **BIPIA has 250 payload strings across 400 payload_ids**, so v18 groups by the
  string, because the ids under-group.
- TF-IDF on P1 reproduced v17c's saved scores to 7e-5 (in-dist) and 1.6e-4 (LOCO), with sklearn 1.8 in the cloud
  against the kernel's version. The integrity tolerance is therefore 1e-3. A misaligned row moves a score by 0.1–1.0.
- **Disclosure:** because the texts are real, the dry run produced **real TF-IDF numbers on P2**: pooled 18.11%
  (P1: 6.07%), BIPIA 13.21 → 51.00, InjecAgent 0.00 → 30.12. This confirms the payload-memorisation diagnosis.
  The pre-registration above was written **before** that dry run and was not changed after it. No real
  MSLSAM-P number has been seen by anyone.


## Results — run 1 (2026-09-28)

Integrity checks passed. TF-IDF and MiniLM on P1/LOCO reproduced v17c **exactly** (max |diff| = 0.00e+00).
Leak asserts passed. Indirect injections: 1,386 rows, **312 distinct payload strings** (payload_id says 462).
Every method's worst own-context FPR is 10.4–10.8%. Runtime about 25 min.

### ASR % at each family's own 10% FPR

| method | protocol | SPML | DeepSet | BIPIA | InjecAgent | pooled |
|---|---|---|---|---|---|---|
| **MSLSAM-P (LR)** | **P2** | **0.17** | **2.73** | 50.30 | **21.89** | **15.81** |
| MSLSAM-P (HGB) | P2 | 0.24 | 1.83 | 54.38 | 34.00 | 17.52 |
| MSLSAM v17 | P2 | 1.15 | 10.52 | **46.75** | 32.63 | 16.35 |
| TF-IDF + LR | P2 | 1.41 | 22.18 | 51.19 | 36.27 | 18.69 |
| MiniLM + LR / HGB | P2 | 5.03 / 1.13 | 49.61 / 36.90 | 74.87 / 84.06 | 72.00 / 84.24 | 31.06 / 31.11 |
| MSLSAM-P (LR) | P2p | 0.22 | 1.86 | 51.13 | 77.83 | 18.53 |
| MSLSAM v17 / TF-IDF | P2p | 1.19 / 1.10 | 7.84 / 20.58 | 48.29 / 50.35 | 75.19 / 72.22 | 18.56 / 19.77 |
| MSLSAM-P (LR) | P1 (leaky) | 0.16 | 2.60 | 26.81 | 14.81 | 8.68 |
| **MSLSAM-P (LR)** | **LOCO** | **30.90** | 40.32 | 73.88 | 83.02 | **46.27** |
| MSLSAM-P (HGB) / v17 | LOCO | 55.04 / 39.96 | 30.66 / 34.19 | 72.61 / 79.70 | 59.22 / 60.78 | 58.76 / 52.02 |
| TF-IDF / MiniLM-LR | LOCO | 86.64 / 77.74 | 55.33 / 71.02 | 80.17 / 79.07 | 74.93 / 75.56 | 82.26 / 77.60 |
| ProtectAI / PG2 / judge | zero-shot | 1.00 / 1.59 / 5.80 | 27.47 / 20.23 / 14.73 | 84.1 / 89.4 / 88.8 | 77.4 / 52.2 / 8.87 | 30.16 / 30.45 / 27.83 |

Paired McNemar, MSLSAM-P (LR) vs TF-IDF: **P2** SPML 32 vs 1 (p = 8e-9), DeepSet 55 vs 4 (p = 2e-12),
InjecAgent 42 vs 16 (p = 9e-4), BIPIA 210 vs 201 (p = 0.69, tie). **P2p** SPML and DeepSet win,
BIPIA and InjecAgent tie. **LOCO** SPML, DeepSet and BIPIA win, InjecAgent ties.

### Pre-registered verdict (mechanical, as printed)

- **H1 bottleneck — does NOT hold.** 15.81 vs 16.35 on P2 (needed ≥ 2). Per family, the probe is much
  better on SPML, DeepSet and InjecAgent, and significantly *worse* on BIPIA (v17's prototypes help there).
- **H2 hidden states beat text — SUPPORTED.** P2 margin 2.88 points; McNemar wins on SPML, DeepSet and InjecAgent, with no losses;
  P2p 18.53 ≤ 19.77, so it is not reversed. By our rule, **v19 is recommended.**
- **H3 transfer retained — HOLDS.** LOCO 46.27, better than v17's 52.02 in this run, and far better than every
  corpus-trained text baseline (77.6–82.3). It is still **worse than the zero-shot guards (27.8–30.5)**.

### Three things that qualify the verdict (none of them changes it)

1. **The PCA seed moves MSLSAM v17 far more than I claimed.** With v17's code and only `random_state=42` added,
   MSLSAM v17 gives P1 15.73% (v17 run 2: 14.42%) and **LOCO 52.02% (v17 run 2: 59.18%)**. So the seed swings it
   ~1.3 points in-distribution and ~7 points in LOCO. My v17 note ("0.2–0.6 pts") was **wrong**. The likely cause is that
   randomized SVD approximates the trailing components of a 512-of-2000 PCA poorly, and Mahalanobis then inverts
   their covariance. v17's verdicts survive, because their gaps are 8+ points, but **H1 (a 0.54-point gap) is inside this noise**,
   and MSLSAM-P's own seed noise has **not** been measured. H2's pooled margin (2.88) is only 0.88 above its threshold.
2. **C hit the edge of the grid.** Inner CV picked C = 0.001, the smallest value, in 18 of 19 splits, and the inner AUC
   falls steadily as C rises. The probe wants more regularisation than the grid allowed, so its numbers are
   probably not its best. That cannot be used after the fact. It is a design input for the next version.
3. **Where the gain is.** It is almost entirely on the **direct** families and InjecAgent. **BIPIA stays at ~50% for every
   trained method**, and InjecAgent collapses to ~75–78% for everyone once P2p starves it of training data. Indirect
   injection is still unsolved, which is exactly what v19's task-drift Δ targets.

### Target (ASR < 5%, 10% FPR budget), payload-held-out split

**Met on both direct families for the first time under a leak-free split:** SPML **0.17% [0.06, 0.41]** at FPR 10.1%,
and DeepSet **2.73% [1.30, 5.39]** at FPR 10.8%. DeepSet is the striking one: every zero-shot guard sits at
14.7–27.5%, and TF-IDF at 22.2%. Caveats that must travel with it:
- This is still *in-family*. The payloads are unseen, but the family was in training, so the fair zero-shot
  comparison remains LOCO, where MSLSAM-P loses.
- DeepSet is n = 263, and its Wilson upper bound is 5.39%.
- Measured FPR is 10.1–10.8%, not < 10%. The budget is 10% on a calibration half. A strict "< 10%" claim needs a
  9% budget.
- Indirect families are nowhere near the target.

### Hard negatives and cost
- Hard-negative FPR, MSLSAM-P (LR), P2: BIPIA 12.0%, **InjecAgent 20.0%** (v17: 6.2%). The probe fires more on
  benign instructions inside tool output.
- **Cost: 0.78 ms marginal** (8 PCA projections + LR), against v17's 3.22 ms. There is no prototype scan, so it no longer
  grows with the size of the training set.


## v18b — seed stability (built and run 2026-09-29)

**Source:** `v18b_seeds.ipynb` (CPU only; about 1 hour with everything on, about 25 min with `RUN_V17 = False`; resumable).
**Why:** v18's H2 passed by 2.88 points against a 2.0 threshold on ONE PCA seed (42), and the PCA seed turned out to
move MSLSAM v17 by 1.3 to 7 points. v18b re-runs MSLSAM-P (LR) on P2, P2p and LOCO, and MSLSAM v17 on P2 and LOCO, with PCA seeds
**0, 1, 2, 3, 4**. Only the PCA `random_state` varies. The folds (P2 asserted identical to v18), C grid, inner-CV split, heads and
thresholds are fixed. TF-IDF is deterministic, so its v18 scores are reused.

### Pre-registered reading (written before any v18b number exists)

For each new seed s ∈ {0, 1, 2, 3, 4}, H2 is evaluated with **exactly v18's rule**. On P2, MSLSAM-P must be ≥ 2 pooled
points below TF-IDF, with McNemar in its favour on ≥ 2 families and against it on none. On P2p, it must not be worse than TF-IDF.

| H2 holds in … of the 5 new seeds | reading | v19 |
|---|---|---|
| **5 or 4** | **H2 is STABLE**, and v18's claim stands | **recommended** |
| 3 or 2 | H2 is **seed-dependent (fragile)**. Report it as such; v18's pass was partly luck | not justified by this evidence |
| 1 or 0 | H2 is **RETRACTED**; seed 42 was a lucky draw | not justified |

Reported as descriptive only; none of these changes the H2 reading:
- **Headline stability:** SPML and DeepSet ASR under P2 for every seed (min–max), and in how many seeds each is < 5%.
- **H1 and H3 across seeds.** If H1 holds in ≥ 4 of 5 seeds, v18's "H1 does not hold" becomes *seed-dependent*. The same applies to H3.
- **The seed spread of each method**: mean, SD and min–max of pooled ASR per protocol. It measures how unstable
  each pipeline is, and it is the noise floor for every earlier MSLSAM comparison.

Not allowed afterwards: adding seeds until a count changes, dropping a seed, or changing the C grid or the folds.


### Dry run (2026-09-29)
- The dry run used the real metadata, texts and folds, plus synthetic 700-dim states, so that randomized PCA is exercised.
- **Integrity:** seed 42 through the v18b code reproduced v18's saved scores **exactly** (max |diff| = 0.00e+00) for both MSLSAM-P and v17.
- Seeds 0 and 1 changed MSLSAM v17's scores by up to 0.99 (the seed really does move it).
- **Caught a trap:** at 700 dims, sklearn ≥ 1.5 resolves MSLSAM-P's PCA to the deterministic `covariance_eigh` solver
  (n_features ≤ 1000 and n_samples ≥ 10 × n_features). That makes every seed identical and the test *vacuous*. On the
  real 4,096-dim states it resolves to `randomized` (checked: a 1,100-dim unit test gave seed-to-seed score differences of 0.30–0.33).
  The notebook now prints, per method, how much each seed changes the scores, and flags "ZERO: the seed has no effect"
  if that ever happens on the real data.
- The dry run's ASR numbers are meaningless (synthetic states).


### v18b results (run 2026-09-29, ~1 h)

Integrity: seed 42 reproduced v18 **exactly** (max |diff| = 0.00e+00) for both MSLSAM-P and MSLSAM v17.
The seed did act: MSLSAM-P's P2 scores moved by up to 0.18–0.21 relative to seed 42 (checked afterwards, see below).

| seed | MSLSAM-P P2 | TF-IDF P2 | margin | McNemar wins / losses | P-P2p vs T-P2p | H2 | SPML | DeepSet | v17 P2 | P LOCO | v17 LOCO |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 42 (v18, not counted) | 15.81 | 18.69 | 2.88 | SPML, DeepSet, InjecAgent / none | 18.53 / 19.77 | ✓ | 0.17 | 2.73 | 16.35 | 46.27 | 52.02 |
| 0 | 15.44 | 18.69 | 3.25 | SPML, DeepSet, InjecAgent / none | 18.37 / 19.77 | ✓ | 0.18 | 2.55 | 16.69 | 47.20 | 58.89 |
| 1 | 15.28 | 18.69 | 3.41 | same | 18.61 / 19.77 | ✓ | 0.16 | 2.70 | 15.67 | 48.02 | 54.65 |
| 2 | 15.60 | 18.69 | 3.09 | same | 18.77 / 19.77 | ✓ | 0.20 | 2.52 | 17.05 | 46.06 | 56.65 |
| 3 | 15.43 | 18.69 | 3.26 | same | 18.20 / 19.77 | ✓ | 0.17 | 2.67 | 17.15 | 46.83 | 58.20 |
| 4 | 15.58 | 18.69 | 3.11 | same | 18.46 / 19.77 | ✓ | 0.16 | 2.56 | 16.30 | 46.75 | 61.76 |

Spread over the 5 counted seeds (mean ± SD [min, max]): MSLSAM-P P2 **15.47 ± 0.13** [15.28, 15.60]; P2p 18.48 ± 0.22;
LOCO 46.97 ± 0.72. MSLSAM v17 P2 16.57 ± 0.61; **v17 LOCO 58.03 ± 2.64 [54.65, 61.76]**. H2 margin **3.22 ± 0.13** [3.09, 3.41].

**Pre-registered verdict: H2 holds in 5 of 5 seeds -> H2 is STABLE, v18's claim stands, and v19 is recommended.**
- **Claimable per-family wins over TF-IDF on P2** (McNemar favoured MSLSAM-P on all 5 seeds): **SPML, DeepSet, InjecAgent**. BIPIA: no claim.
- **Target:** SPML 0.16–0.20% and DeepSet 2.52–2.70% across seeds, so the worst seed is < 5% for both, and the claim is allowed.
  The DeepSet caveats remain: n = 263, Wilson upper bound 5.39%, measured FPR 10.65–10.79%.
- Descriptive: H1 holds in 0/5 seeds, so "the 24-feature summary was not the bottleneck" is now firm, on pooled numbers. H3 holds in 5/5 seeds.
- **The probe is ~5x more seed-stable than v17** (P2 SD 0.13 vs 0.61; LOCO 0.72 vs 2.64). Seed 42 was v17's *luckiest* LOCO
  draw (52.02, below all five other seeds). So v18's H3 comparison *understated* the probe's transfer gain: it is about 11 points
  on the seed mean (46.97 vs 58.03), not 5.75.
- Seed 42 was MSLSAM-P's *worst* P2 margin of the six. v18 did not get lucky.

**Execution note:** the notebook that ran did not contain the "score diff per seed" safeguard I added after the dry run.
The file transfer delivered an earlier copy. The check was done afterwards on the saved scores: every seed changed
MSLSAM-P's P2 scores by 0.18–0.21, so the test was not vacuous.
