# MSLSAM — reading prompt injection from inside the model

Undergraduate thesis research (Skripsi), Computer Science, BINUS University Malang, 2026.
**Authors:** Stanley Pratama Teguh, Kezia Karen Amelia.

## What this is
- **v01–v25:** MSLSAM, a lightweight detector that reads an LLM's own hidden states (Llama-3-8B-Instruct) during the forward pass the
  application already runs. Evaluated under increasingly strict, pre-registered protocols: grouped CV, per-source false positives,
  held-out attack strings, external-only training, frozen models on fresh data.
- **v26 onward (current):** *when the attacker moves second, which internal signals still stop a prompt injection that actually works?*
  An independent adaptive evaluation of internal-signal detectors (presence probes, attention routing, breach probes) on
  Qwen3-4B-Instruct and Llama-3.2-3B-Instruct, with success checked on real generations.

Every version has a pre-registration, written before it ran, and an honest result, negative results included.

## Where to look
| path | what |
|---|---|
| `EXPERIMENTS.md` | master table of every version and its result |
| `versions/NN_name/README.md` | per version: design, pre-registration, run log, result |
| `versions/NN_name/code/` | the notebooks exactly as they were run |
| `results/vNN/` | metrics, scores and audit files per version |
| `outputs/` | method write-up and recaps |

## Reproducing
- Python 3.11, PyTorch with MPS (developed on a 24 GB Apple-silicon MacBook), `transformers` ≥ 4.56, scikit-learn, pandas.
- Hugging Face access to the gated Llama models. Authenticate with `hf auth login` or the `HF_TOKEN` environment variable.
  **Never put a token in a notebook.**
- Run order and time estimates for each version are in its `versions/.../README.md` (run notebooks from this folder). Notebooks print a swap check and a smoke-test
  estimate before any long run.

## Not in this repository
- **No datasets.** `representation_cache/` (~27 GB of cached hidden states) and `datasets/` (~1.3 GB) stay local, as does any file that
  copies dataset text. The notebooks rebuild them. The datasets come from public sources under their own licences: deepset prompt-injections, SPML, BIPIA, InjecAgent,
  WildJailbreak, Dolly-15k, HarmBench, JailbreakBench, XSTest, Alpaca, Enron ham, Gandalf / Lakera sets, glaive function-calling,
  IMDB, LLMail-Inject, NotInject, oasst1, SQuAD, Tensor Trust, CNN/DailyMail, XSum, WikiTableQuestions and no_robots.
- Data and model file types (`.parquet`, `.npy`, `.npz`, `.pt`, `.safetensors`, archives, …), except the small frozen v26 detectors
  in `artifacts/v26/` that later versions verify by SHA-256.
- The thesis documents, papers and slides, which live outside this folder.

## Licence
No licence has been chosen yet, so all rights are reserved by default.
