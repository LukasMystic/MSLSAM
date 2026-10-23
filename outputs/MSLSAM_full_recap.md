# MSLSAM: recap of the whole experiment series (v01 → v21), written 2026-10-12

**Question:** can the hidden states of the LLM being protected (Llama-3-8B-Instruct) detect prompt injection, direct and indirect, at a
useful operating point?

**Target:** attack success rate (ASR) < 5% with false-positive rate (FPR) < 10%.

**Status:** v21 confirmation pending (v21b2 → v21c not run yet). Source of truth: `EXPERIMENTS.md`, `outputs/MSLSAM_methods.md`, `versions/*/README.md`.

---

## 1. The arc in five phases

### Phase A — finding a signal (v01–v09, Mar–Jun 2026)
| ver | idea | result |
|---|---|---|
| v01 | GPT-2: attention entropy + trajectory | null (ASR 75%) |
| v02 | Llama-3-8B, PCA → Mahalanobis + refusal direction | **AUC 0.95**, ASR 8.5%, FPR 12.8%: the best classical result |
| v03 | layer sweep + gray zone | gray zone too wide |
| v3–v6 | — | **lost**: overwritten in place in one notebook cell |
| v2 (MSLSAM), v7, v8 | batching/caching, MoE forest, stratified thresholds | infrastructure; v8's key number was never recovered |
| IDP-CJ | injection direction + constitutional judge | ceiling, TPR plateaus at 0.88 |
| v9 | 26 trajectory features + LR | **score collapse** (AUC 0.83) |

*Takeaway:* the representation carries a signal (layer-4 margin against adversarial-benign prompts is ~3× any other depth), but the classifiers
and bookkeeping were not yet trustworthy.

### Phase B — calibration and the first "target met" (v10–v13, 15 Sep)
- **v10:** prototype max-cosine. Ranking fixed (injection AUC 0.92), target missed.
- **v11:** calibrated FPR budget, all 8 layers. **ASR 4.18% at 10% FPR: target met, in-domain.**
- **v12:** grouped CV + nested thresholds. **ASR 3.20%, AUC 0.985.**
- **v13 overturned v12.** Re-measuring FPR per source showed adversarial-benign prompts at **99.7% FPR**. No configuration meets both targets.

*Takeaway:* a pooled FPR can hide a context where the detector fires on almost everything. **Every benign context must be measured separately.**

### Phase C — a realistic corpus, including indirect injection (v14–v16, 27–28 Sep)
- **v14:** unified corpus (SPML, DeepSet, BIPIA, InjecAgent, plus hard benign sets), payload-grouped CV, length-matched.
  - It **solved the adversarial-benign confound** (WJB-Adv-Benign FPR 7.49%).
  - Its headline ASR of 3.25% is **not quotable**: the pooled FPR hid InjecAgent-Clean at 100%.
- **v15:** position-relative chunk features. **Premise falsified.**
  - The payload is not a localised jump in the trajectory; the signal is a whole-prompt property.
  - Hard negatives (benign instructions inside a carrier) reach 41–52% FPR.
- **v16:** hard-negative training.
  - **SPML ASR 0.91% at FPR 9.91%: target met** (direct, in-distribution).
  - Indirect injection was not solved. Hard-negative training traded 47 points of InjecAgent detection for 22 points of FPR.
- **The binding constraint was named here:** each benign context needs its own threshold, and those thresholds span **5.4 × 10³**
  (from 0.0002 for short questions to 0.97 for retrieved documents).

### Phase D — honest comparison (v17–v18b, 28–29 Sep)
- **v17:** 12 methods on identical folds. **Both pre-registered verdicts went against MSLSAM.**
  - TF-IDF + LR trained on the same corpus beat it in-distribution (6.07 vs 14.42), so the lead came from the **training data**, not the representation.
  - The zero-shot guards (~28–30% pooled) beat MSLSAM held out (59%).
  - What survived: MSLSAM transfers to unseen families far better than any baseline trained on the same corpus (59 vs 78–82).
  - A post-hoc finding: the in-distribution folds **leaked payload strings** on BIPIA and InjecAgent (96–98%).
- **v18:** MSLSAM-P, a linear probe directly on the hidden states, evaluated with a **payload-held-out split** (P2).
  - **H2 supported:** hidden states beat text, with 3 McNemar wins and 0 losses.
  - SPML 0.17%, DeepSet 2.73% (target met on both, direct).
  - BIPIA ~50% for every method.
- **v18b:** H2 is **stable in 5/5 seeds** (margin 3.22 ± 0.13). It also exposed that PCA seed noise was larger than I had claimed (~7 points held out).

### Phase E — fair training, then confirmation (v19–v21, 29 Sep → now)
- **v19:** every trained method learns **only from external data** (5,916 rows), and a task-drift feature Δ = h_end − h_anchor is added.
  - MSLSAM **matches** the guards pooled (29.5 vs 28.4), with complementary strengths: better on BIPIA (62 vs 82), worse on InjecAgent (82 vs 53).
  - **SPML is solved by length alone** (AUC 0.994), so SPML results carry no claim.
- **v20a:** z-max ensemble with PG2. **Failed (0/5).** On tool output, MSLSAM's score drifted +3.6 SD and dominated the max.
- **v20b:** +1,500 external tool-output rows.
  - InjecAgent 82 → 55 (T1 holds).
  - BIPIA +3.4 (T2 fails).
  - **On the length-clean indirect families, MSLSAM beats PG2 in 5/5 seeds (63.9 vs 78.2).**
  - Cost: tool hard-negative FPR 14 → 26%.
- **v21:** the frozen v20b detector, tested once on 1,800 fresh rows (news summarization; BIPIA attacks plus never-seen Tensor Trust attacks).
  - Integrity passed: the refit equals v20b to 3e-08, and the deployable numpy scorer equals sklearn.
  - The Llama latency benchmark was **abandoned** (the machine was swapping 50 GB).
  - **C1/C2 not yet computed.** Details: `outputs/v21_recap.md`.

---

## 2. Where the target stands
| setting | met? |
|---|---|
| Direct, in-distribution (v11, v12, v16 SPML 0.91%) | yes, but the v11/v12 numbers were overturned (v13), and SPML turned out to be length-trivial (v19) |
| Direct, payload-held-out, corpus-trained (v18: SPML 0.17%, DeepSet 2.73%) | yes, with the SPML caveat; DeepSet is the meaningful one |
| Any family under **fair** (external-only) training (v19–v20b) | **no** |
| Indirect injection (BIPIA, InjecAgent), any version | **no**. The best is ~55–65% ASR; the 8B judge reaches 3.9% on InjecAgent |

## 3. What the thesis can claim today (before v21's verdict)
1. Llama's hidden states carry injection signal **beyond surface text** when both are trained on the same data (v18 H2, stable 5/5).
2. Under fair training, MSLSAM is **on par with dedicated guards** pooled, and **complementary** per family (v19).
3. With tool-output training data, it beat PG2 on the length-clean indirect families in v20b. **This needs v21 to confirm it on unseen data.**
4. Its marginal cost is **< 1 ms** on top of a forward pass the application already runs (0.78–0.81 ms vs PG2's 32 ms and the judge's 378 ms).
5. **Negative results worth reporting:**
   - the signal is not localised (v15)
   - hard-negative training is a trade, not a fix (v16)
   - thresholds must be set per context (5.4 × 10³ spread)
   - z-max ensembles break under calibration drift (v20a)
   - payload leakage in carrier-grouped splits (v17)

**Not claimable:**
- the target in general
- indirect-injection defence
- robustness to an adaptive attacker
- other models or other system prompts

## 4. What went right (method)
Rigor was added step by step, and each step overturned something:
- grouped CV (v12)
- per-source FPR (v13)
- context-matched thresholds (v14/v16)
- external baselines on identical folds (v17)
- payload-held-out splits (v18)
- seed stability (v18b)
- pre-registration (v17 on)
- length controls (v19)
- external-only training (v19)
- fresh-data confirmation plus freeze/reproduce (v21)

**The project kept its negative results instead of burying them.** That is the thesis's strongest asset.

## 5. What went wrong (recurring)
- **Lost and unrecoverable work:** v3–v6 overwritten, v8's number never recovered, a diagnostics notebook mostly unrun. → The rule now is that every version gets its own folder and frozen code.
- **Headlines announced, then overturned:** v12 by v13; v14's 3.25% and 7.09%. → The pooled metric hid per-context failures. Report per family, never pooled alone.
- **Leakage and confounds found late:** payload reuse across folds (v17), length (v19), test-set overuse (v17–v20b).
- **My errors along the way:**
  - I understated PCA seed noise.
  - A stale notebook copy was committed (v18b).
  - I miscounted Gandalf rows (v19a).
  - A json-escaping bug crashed v20b1.
  - I designed v21b to co-load models on a 24 GB Mac and ignored early signs of swapping.

  Each produced a convention in AGENTS.md: grep the committed file, keep dry-run data realistic, checkpoint per item, check swap before Llama runs.
- **Environment:** a 24 GB Mac running an 8B model is at the edge. Swap destroyed v21b. The Hugging Face token in `main.ipynb` cell 1 is still to be revoked.

## 6. Recurring insights
1. **Calibration per context is the real problem.** The detector ranks well (AUC 0.95–0.99 in-domain), but no single threshold fits documents,
   tool output and short questions at once.
2. **Data beats features.** Most in-domain gains came from training on the same families (v17). The fair setup (v19) is the honest one.
3. **Indirect injection is a different problem.** The payload sits inside legitimate content. Detecting "an instruction is present" fires on benign
   instructions too: hard negatives run at 26–52% FPR.
4. **Cheap and complementary.** MSLSAM's value is near-zero marginal cost and errors that differ from the guards'. That is the case for an ensemble, if v21's C2 supports it.

## 7. Open before the write-up
- Restart the Mac → run v21b2_guards → run v21c (C1, C2, deployment view, cost table).
- Decide what the thesis headline is after C1/C2.
- Not covered by any experiment: an adaptive attacker, a second model, varied system prompts, the annotation pilot (kappa).
- Revoke the HF token.

---
## Update 2026-10-18 — v21 ran
- **C1 holds in 5/5 seeds.** On 1,800 fresh rows the frozen v20b MSLSAM-Δ beats Prompt Guard 2: FRESH pooled ASR 11.6 vs 38.1; Sum-TT (never-used payloads) 5.5 vs 8.8.
- **C2 fails (0/5):** the z-mean ensemble is not worth its cost overall.
- **Target:** reached only by the ensemble on Sum-TT (3.3%, Wilson upper bound 6.0%, FPR 10.5%), so borderline.
- **Costs that go with the claim:**
  - harmless instructions inside documents are flagged 37–71% of the time (PG2 10–14%)
  - thresholds set on external data give 28–71% FPR on clean news
- Details: `versions/20_mslsam-v21/README.md`.
