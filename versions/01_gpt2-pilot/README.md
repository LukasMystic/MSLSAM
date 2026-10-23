# 01 — GPT-2 Pilot

**Run:** 2026-03-08 · **Source:** `main.ipynb` cell [2] · **Status:** ❌ null result

## Design
Attention entropy + hidden-state trajectory distance from a baseline "anchor" built on
500 Dolly-15k prompts. Fixed detection threshold 0.6. Model: `gpt2`, device MPS.

## Data
| Set | N |
|---|---|
| Baseline anchor (Dolly-15k) | 500 |
| Safe edge cases (XSTest-Safe, JBB-Benign, WJB benign) | 1350 |
| Attack prompts (HarmBench, JBB, XSTest-Unsafe, WJB harmful) | 1684 |

Preprocessing removed 3 HarmBench-Test and 2 HarmBench-Val duplicates; all other sets clean.

## Results
| Metric | Value |
|---|---|
| Threshold | 0.6 (fixed) |
| FPR | **25.33%** |
| ASR | **75.06%** |

## Verdict
Barely better than chance. Value was infrastructural: it established the dataset loader,
preprocessing, deduplication and EDA that every later version reuses. GPT-2 is too small
and has no instruction hierarchy to violate, so there is no injection signal to find.

Abandoned in favour of Llama-3-8B-Instruct (version 02).

## EDA artifacts
Prompt-length statistics worth keeping: WJB-Adv-Harmful averages 153.3 words vs 16.6 for
WJB-Vanilla-Harmful, and WJB-Adv-Benign 128.3 vs 12.8 vanilla. **Adversarial prompts are
~10× longer than vanilla ones in both classes** — a confound to control for, since any
length-sensitive feature will separate adversarial from vanilla regardless of intent.
