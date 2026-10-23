# v3 – v6 — LOST

**Status:** 🚫 unrecoverable

Between version 03 (2026-04-20) and version 04 / MSLSAM v2 (2026-06-09) the owner's
version numbering ran through v3, v4, v5 and v6. **No trace of these remains.**

## Why
All of them lived in `main.ipynb` cell [6], which was edited in place across roughly two
months. Each version overwrote the previous one. The notebook has no saved outputs for
that cell, the figures were overwritten by later runs, and the project is not under
version control (AGENTS.md R6).

What survives from that stretch is only what later cells reference: the v7 MoE classifier
and the `v8` feature vector (16-dim, 8 Mahalanobis + 8 refusal-direction over
L4/8/12/16/20/24/28/31), both of which appear in the cell [6] snapshot archived under
`../06_mslsam-v7/` and `../07_mslsam-v8/`.

## Consequence
The methodology chapter cannot narrate a continuous v1→v10 progression. Either present the
lineage as the seven recoverable versions, or state plainly that intermediate iterations
were not preserved.

## This is the reason for AGENTS.md R1 and R3
Snapshot the code and write down the result **at the time of the run**. Without git, the
folder structure is the only history there is.
