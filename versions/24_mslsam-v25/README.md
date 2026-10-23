# 24 — MSLSAM v25 (single layer vs eight; which layers matter; AUC vs deployability; cost; how much calibration a new context needs)

**Status:** ✅ RUN 2026-10-19 — **B: WORSE (5/5): a single layer-12 probe beats the 8-layer head on 6 fresh families (5.5 vs 12.3 pooled ASR); D: AUC tracks ASR (ρ −0.99) but saturates; E: released TaskTracker costs +138–147% of the prefill; F: ~100 clean docs per context.** Built 2026-10-19 (pre-registration below unchanged).

## Why
The closest prior work is TaskTracker (Abdelnabi et al., 2024). It uses the same signal: a linear probe on the hidden-state difference
between "task only" and "task + data", on Llama-3-8B among other models. It reports ROC AUC only, and its headline configuration is a
**single-layer** logistic regression on the raw 4,096-dim delta. v25 asks five cheap questions that place MSLSAM against it, and against
deployment. **It is CPU-only: no Llama, no guard model, no download.** It uses the hidden-state caches and scores that already exist.

## Data
- **Training:** v20b's training set, unchanged (v19 external + v20 tool; 7,416 rows; 8 layers).
- **Development families (selection only):** BIPIA and InjecAgent (v19 test), with their Clean tier, plus Reader (v22) and HardNeg tiers.
- **Fresh families (comparison):**
  - Sum-BIPIA, Sum-TT (v21)
  - Review-TTX, Table-TTX (v23)
  - News-GS, Table-MC (v24)
  - each with its Clean, Reader and AI-addressed off-task tiers (HardNeg / OffTask)
  - They have all been used once, but **v20b was frozen before any of them existed**. No method compared here was fitted or selected on them.
- **Protocol:** v17b, unchanged. Each family is scored at its own 10% FPR on its Clean tier; Reader and off-task are extra rates at that threshold.

## The tests
**Integrity (I1):** v20b's recipe refit on all 8 layers must reproduce the frozen A0 (< 1e-4). A0 must reproduce its saved v22 / v23 / v24
scores, and ENS0 its v24 scores.

**B. Single layer vs eight.** Both variants use the same training rows.
- **B1 = TaskTracker's probe type:** `LogisticRegression()` with C = 1 on the **raw** 4,096-dim Δ of one layer. No PCA, no scaler, no class
  weights; this is TaskTracker's `train_linear_model.py`, with max_iter raised for convergence. One fit per layer, 8 layers.
- **B2 = MSLSAM's recipe on one layer:** PCA(256), scaler, LR balanced, C by inner grouped CV. Seeds 0–4. Only the number of layers differs
  from A0.
- **Selection, on development data only:** B1* (and B2*) is the layer with the lowest pooled development ASR (BIPIA + InjecAgent).
- **Comparison on the 6 fresh families, pooled, per seed** (A0_s vs B1*, and A0_s vs B2*_s):
  - **BETTER:** A0 ≥ 2 points lower, at least 1 McNemar win and no McNemar loss
  - **EQUIVALENT:** under 2 points either way
  - **WORSE:** A0 ≥ 2 points higher
  - **MIXED:** otherwise
  - The verdict is the category that occurs in ≥ 4 of 5 seeds; otherwise SEED-DEPENDENT.
- **Pre-registered reading:**
  - BETTER → combining 8 layers adds detection over TaskTracker's single-layer configuration.
  - EQUIVALENT → the multi-layer head is **not** a contribution; one layer suffices.
  - WORSE → the multi-layer design is a negative result.
- **Descriptive only:** the best single layer *per fresh family* (an optimistic upper bound for single layers), and Reader FPR per layer.
- **Caveat:** this is TaskTracker's *configuration* on our single-pass, unprimed Δ and our data. It is not the released TaskTracker; that
  needs a GPU run and is out of scope here.

**C. Which layers matter (descriptive).** v20b's recipe is refit on:
- each leave-one-layer-out set (8 sets)
- early {4, 8, 12}, middle {16, 20, 24} and late {28, 31}

Seeds 0–4. Reported: development and fresh pooled ASR and Reader FPR, each against all 8 layers.

**D. Does AUC predict deployability?** For every method × family cell (seed means), AUC (injected vs Clean) is plotted against ASR at 10% FPR.
- Methods: A0, ENS0, B1, B2, C, PG2, ProtectAI.
- **Pre-registered statistic:** the share of cells with AUC ≥ 0.95 whose ASR exceeds 10%, plus the Spearman ρ between AUC and ASR.

**E. Deployment cost of the released TaskTracker configuration.**
- TaskTracker's released probes were trained on a *primed* prompt: their system prompt plus "here are your main requests: <MAIN> … </MAIN>
  but first before you answer, please complete the following sentence…". That differs from the app's own prompt, so deployment needs **two
  extra prefills**: task only, and task + data.
- **Counted for the fresh rows** with the Llama tokenizer and v23d's counted FLOP model.
- **Framing:** TaskTracker's own Appendix D.3 notes that an unprimed variant could read the app's pass for free. **MSLSAM is that
  single-pass variant**, so E reports the cost of the *released* configuration, not of the idea.

**F. How many clean documents does a new context need for calibration?** For each fresh family and each n in {10, 25, 50, 100, half of
Clean}, 400 draws are made:
- the threshold is set on n random Clean rows
- FPR is measured on the remaining Clean rows, and ASR on the injected rows
- Methods: A0 (seeds 0–4), ENS0 (seeds 0–4), PG2, ProtectAI.
- **Pre-registered rule:** the recommended n is the smallest n whose worst-family, worst-seed 95th-percentile FPR is **within 2 points of
  the value at n = half**. In plain words: more clean documents no longer help.
  - The absolute numbers are reported as well.
  - An absolute cap (such as "p95 ≤ 15%") was rejected before the run. FPR measured on about 110–150 held-out documents is itself noisy:
    a binomial p95 of about 14–15% at a true 10%. A cap would therefore measure evaluation noise, not calibration size. The dry run showed this.

**Cannot be changed after the run:** training data, recipes, seeds, layer sets, the selection rule, the categories, the statistic, the n grid,
the rule for n.

## How to run
| step | notebook | where | time |
|---|---|---|---|
| 1 | `v25_layers_calibration.ipynb` | CPU, any kernel | ~45–75 min (about 100 fits; resumable per fit) |

## RESULT (run 2026-10-19) — B: the 8-layer head LOSES to a single layer (layer 12), pre-registered, 5/5 seeds

### Run log
- One CPU notebook, all cells OK.
- **I1 passed:**
  - A0 reproduces v22, v23 and v24 (< 1e-4); the frozen ENS0 reproduces v24 (< 1e-6).
  - v20b's recipe refit with v20b's own C is identical to the frozen A0.
- 120 score columns, all evaluated in 14 s.

### B — single layer vs eight (pre-registered)
- **Selection on development data (BIPIA + InjecAgent): B1* = layer 12 and B2* = layer 12.** Development pooled ASR: B1 L12 34.5,
  B2 L12 37.4, A0 63.9.
- **Fresh families** (pooled over 6; ASR % at each family's 10% FPR):

| | pooled | Sum-BIPIA | Sum-TT | Review-TTX | Table-TTX | News-GS | Table-MC | Reader FPR |
|---|---|---|---|---|---|---|---|---|
| A0 (8 layers, frozen v20b) | 12.28 | 17.7 | 5.5 | 2.6 | 14.8 | 3.0 | 24.6 | 28.9 |
| **B1 L12 (TaskTracker probe type, raw Δ)** | **5.48** | 8.2 | 1.7 | 1.4 | 2.6 | 0.9 | 15.2 | 36.2 |
| B2 L12 (our recipe, 1 layer) | 8.92 | 14.1 | 2.3 | 1.7 | 4.4 | 0.9 | 24.8 | 25.8 |
| ENS0 (A0 + PG2) | 7.48 | 19.8 | 3.1 | 1.1 | 0.7 | 0.8 | 14.5 | 25.4 |
| Prompt Guard 2 | 25.34 | 67.3 | 8.8 | 6.6 | 4.8 | 16.1 | 36.7 | 11.8 |

- **Verdict: A0 vs B1* = WORSE (5/5), and A0 vs B2* = WORSE (5/5).**
  - A0 loses McNemar on 4–5 of the 6 families in every seed and wins on none.
  - **Pre-registered reading: the multi-layer design is a negative result.** A single mid-layer probe is better.
- **The trade-off:**
  - Layer 12 has the highest Reader FPR of any probe (B1 36%), so it detects "instruction present" most aggressively.
  - Deeper layers (20–31) have the lowest Reader FPR (18–20%) but 20–28% ASR.
- **Post hoc** (`code/posthoc_v25_reader_calibration.py`): with the threshold set on Clean + Reader, so that reader instructions count
  as benign, B1 L12 still wins.

  | | pooled ASR |
  |---|---|
  | **B1 L12** | **10.1** |
  | A0 | 18.2 |
  | ENS0 | 9.7 |
  | PG2 | 26.2 |
  | ProtectAI | 26.4 |

  The cheap single-layer probe is about as good as the ensemble, at 0.0003% of the prefill instead of 1.8%.

### C — which layers matter (descriptive; fresh pooled ASR)
- Leave-one-out changes little (12.2–14.8): no single layer is essential to the 8-layer head. Removing layer 8 hurts most (14.8).
- **Early {4, 8, 12} = 9.9**, better than all 8 (12.3). **Middle {16, 20, 24} = 24.0. Late {28, 31} = 23.3.**
- The signal for attacks sits in layers 8–16, peaking at layer 12. Adding the deeper layers dilutes it, presumably through the shared PCA,
  scaler and regularisation.
- **Reader FPR falls with depth** (layer 8: 40% → layer 31: 18–20%), so the same layers that catch attacks best also flag reader instructions most.

### D — AUC vs deployability
- **The pre-registered statistic:** 92 of 248 cells have AUC ≥ 0.95, and 13% of those still let more than 10% of attacks through.
  Spearman ρ(AUC, ASR) = −0.99.
- **AUC and ASR rank detectors almost identically**, so the strong claim "AUC misleads" is **not** supported.
- **What is supported: AUC saturates.**

  | AUC | ASR range (median) |
  |---|---|
  | 0.95–0.97 | 6.6–13.8% (9.3) |
  | 0.97–0.99 | 2.3–7.3% (4.3) |
  | ≥ 0.99 | 0.7–3.2% (1.4) |

  A 0.02 change in AUC near the top is a 5–10× change in missed attacks. ASR at a fixed FPR must therefore be reported; AUC alone cannot
  show whether a target is met.

### E — cost of the released TaskTracker configuration
- Its primed prompt needs 2 extra prefills: **7.7–8.8 TFLOPs per prompt, or 138–147% of the app's own prefill.**
- MSLSAM's head costs 0.0168 GFLOPs, or 0.0003%.
- Framing, as pre-registered: MSLSAM, and the B1 L12 probe in the same single-pass setting, is the unprimed variant that TaskTracker's
  Appendix D.3 only proposes.

### F — calibration size for a new context (400 draws; worst family, worst seed)
- The **pre-registered rule gives n = 100** clean documents for every method (A0, ENS0, PG2, ProtectAI).
- The median FPR is already about 10.4% at n = 25, but the worst-case 95th percentile is 24–25% at n = 25 and 19–20% at n = 100.
- The ~18% floor at n = half is the noise of measuring FPR on about 110–150 held-out documents.
- **Recommendation:** about 100 clean documents per deployment context.

### What this changes for the thesis
1. **The 8-layer MSLSAM head is not the best detector.** A single-pass, single mid-layer (12) logistic probe on the raw Δ is better on fresh
   data (pre-registered, 5/5 seeds). That probe type is TaskTracker's. The thesis must report this, and should name the layer-12 probe as the
   best configuration found.
2. **The contribution is the evaluation, not the architecture:**
   - fixed-FPR ASR on 3 fresh test sets
   - head-to-head against production guards
   - the failure map (reader instructions, formats, calibration)
   - the cost analysis
   - the layer analysis (a mid-layer peak, and a depth trade-off between detection and reader false alarms)
   - the pre-registered ensemble result
3. **Open, post hoc only:** a layer-12 + PG2 ensemble. It would need out-of-fold references, and there is no fresh data left to confirm it.

