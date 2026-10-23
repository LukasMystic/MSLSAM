# versions/

Per-version archive. See `../EXPERIMENTS.md` for the master summary and
`../../AGENTS.md` for the rules that govern this folder.

## Contents
| Folder | Version | Result |
|---|---|---|
| `01_gpt2-pilot/` | GPT-2 pilot | FPR 25.3%, ASR 75.1% |
| `02_llama-baseline/` | Llama + Mahalanobis/refusal | AUC 0.9495 |
| `03_layer-sweep/` | Layer sweep + gray zone | AUC 0.9340 |
| `_lost_v3-v6/` | v3–v6 | unrecoverable |
| `04_mslsam-v2/` | MSLSAM v2 | AUC 0.894 |
| `05_idp-cj/` | IDP-CJ | AUC 0.871 |
| `06_mslsam-v7/` | MSLSAM v7 MoE | not recorded |
| `07_mslsam-v8/` | MSLSAM v8 | global AUC 0.944 |
| `08_mslsam-v9/` | MSLSAM v9 trajectory | AUC 0.831, collapsed |
| `09_mslsam-v10/` | MSLSAM v10 prototype | never run |
| `_diagnostics/` | Cells 9–13 | only cell 13 ran |

## Structure
Each folder holds `README.md` (design, config, results, verdict), `code/` (snapshot of the
notebook cell that produced it) and `figures/` (copies of that version's figures).

**Numbering:** `NN` is chronological by run date; the name is the owner's own version label.
They do not match — there is no v3–v6 — so both are kept. `07_mslsam-v8` is the 7th run and
is called v8.

## Adding a version
1. `mkdir -p versions/NN_name/{code,figures}`
2. Snapshot the notebook cell(s) into `code/`
3. **Copy** (never move) the figures into `figures/`
4. Write `README.md`: design → config → results → verdict, bad results included
5. Update `../EXPERIMENTS.md` and `../../AGENTS.md`
