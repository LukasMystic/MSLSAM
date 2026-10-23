## Build notes (2026-10-22, written BEFORE any v27 run)
- **Notebooks:** `Experiments/v27a0_freeze`, `v27a_blackbox`, `v27b_whitebox`, `v27c_evaluate`, plus the shared module `Experiments/v27lib.py`
  (success rule, `thr`, the frozen detectors in numpy and torch, and v26b's prompt/hook pipeline). Copies are in `code/`.
- **Dry run (cloud):** all four notebooks ran on toy tokenizers and tiny random Qwen3 / Llama models.
  - **v27a0:** I6 reproduced v26's AUROCs to 1e-6 for all 5 detectors on both models.
  - **v27a:** F0 (live scores and response reproduce v26b's cached row), smoke gate → N, resume, I7, I8.
  - **v27b:** F1 (KV-cached forward == full forward), F2 (the differentiable detector score == the frozen numpy score on the same
    prompt), F3 (finite gradient). The GCG loss falls, and the targeted detector score moves toward evasion.
  - **v27c:** synthetic records for "routing holds", "all fall" and "weak attacker" give the pre-registered decisions.
- **Change made before the build (budget):** as pre-registered, stage B with 8 runs per instance would need about 40 min per instance
  (N ≈ 18 < 24). Now **every instance gets the 4 primary runs (C0, and C-D3 / C-D4 / C-D6 at λ = 1)**, and the λ = 0.3 / 3 frontier
  runs only on the first 12 instances (descriptive). The budget rule counts both.
- **Stage A size:** population 6, then 7 generations of (best 2 + 4 new), so **at most 34 evaluated candidates** (the README's "≤ 48"
  was an upper bound). The seed is evaluated once per instance and reused across conditions (greedy, deterministic).
  The rewriter's sampling seed is fixed per (instance, condition, generation, slot).
- **Smoke gates:** A times the first two instance-conditions and extrapolates the per-candidate cost to the worst case (34 candidates ×
  4 conditions). B times 3 GCG steps + 1 verification and extrapolates to 150 steps + 16 verifications per run. N is written to
  `results/v27/<model>_<stage>_N.json` and never recomputed.
- **Suffix init:** 20 copies of the first single-token filler (" !", " x", " a", …) that re-tokenises to itself in the prompt; the
  choice is printed. Candidate tokens: printable ASCII, not special. Each candidate must re-tokenise to itself (standard GCG filter).
- **KV cache:** `crop(-k)` (removes k tokens) works in transformers 4.x and 5.x. Each gradient step uses a fresh copy of the prefix
  cache (a backward pass leaves graph-linked slices behind).
- **Last layer:** the probes' layers are mid-depth (Qwen 17 / 21, Llama 13 / 14), so `hidden_states[li+1]` is the pre-norm residual,
  as in v26b; F2 would catch a mismatch.
- **v27c:** a win that I7 could not reproduce counts as a loss. Instances are compared only where all four primary conditions ran.
