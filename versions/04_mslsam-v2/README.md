# 04 — MSLSAM v2 (Batch Extraction + Representation Caching)

**Run:** 2026-06-09 · **Source:** `main.ipynb` cell [6] · **Status:** ⚙️ infrastructure release

## Design
Detection algorithm **identical to version 02** (Mahalanobis + refusal direction). This
release is about making the pipeline fast enough to iterate on. Hooks widened to the full
set **L4, 8, 12, 16, 20, 24, 28, 31**, PCA 128.

### Performance work
| ID | Change |
|---|---|
| PERF-1 | Batched forward passes (`batch_size=4`), **left-padding** so position[-1] is always the real last token |
| PERF-2 | Call `LlamaModel` body directly, skip `lm_head` (~100–130 MB/pass); `use_cache=False` (~67 MB) — together this removed the MPS OOM |
| PERF-3 | Disk caching of hidden states to `representation_cache/`; later runs with same model + layers + system prompt skip the model entirely |
| PERF-4 | Vectorized batch scoring — matrix ops over all N instead of a Python loop |

### Accuracy fixes
| ID | Change |
|---|---|
| FIX-1 | `topic_damping` 0.5 → **0.75** (targets JBB-Benign false positives) |
| FIX-2 | Full JBB-Benign used for the sensitive-safe pool (no `:50` cap) |
| FIX-3 | Phase-3 diverse safe subset: 10 Dolly + 30 JBB-Benign |
| FIX-4 | Stratified calibration ~35 per attack source (~140 total) |

## Results
| Metric | Value |
|---|---|
| ROC-AUC | **0.894** |

Figures: `v2_roc_curve.png`, `v2_pr_curve.png`, `v2_confusion_matrix.png`,
`v2_score_distribution.png`, `v2_per_source_boxplot.png`.

## Verdict
AUC fell from 0.9495 to 0.894 versus version 02 — but the evaluation set, layer set and
damping all changed at once, so the drop is not attributable. **The lasting contribution is
`representation_cache/`**: ~45 `.npz` files holding every dataset's hidden states at all 8
layers, 4096-d. Every subsequent analysis can run without loading the model, which is what
makes the diagnostic cells cheap to execute.

Raising damping to 0.75 did not fix JBB-Benign; it remains the worst FP source through v8.
