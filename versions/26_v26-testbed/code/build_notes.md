
## Build notes (2026-10-20, written BEFORE any v26 run; implementation details the pre-registration left open)
- **Notebooks:** `Experiments/v26a_build`, `v26b_extract_generate`, `v26c_audit`, `v26d_fit_freeze`, `v26e_evaluate` (copies in `code/`).
  All five passed a dry run in the cloud workspace on toy byte-level tokenizers (Llama-3-style and Qwen-style chat templates) and tiny
  random Llama / Qwen3 models. The dry run covered:
  - fidelity: hooks equal `output_hidden_states` / `output_attentions`, and generation is unchanged by the hooks
  - resume after a deleted shard, and the 8B check-model path (test + selection rows only)
  - synthetic success labels and planted signals on **both gate branches** (PASS and FAIL), the V2 stop path, and the I5 stop path
- **I1, exact rule:** outside the inserted span, the attack and its twin must have identical token ids. The boundary excluded from the
  comparison is the token touching the span's start and **the first word after the span**: a word that gains or loses its leading space
  tokenises differently, and the tokens re-synchronise at the next whitespace. Checked in v26a with every tokenizer that loads.
- **Carrier caps:** leading sentences up to 2,000 characters (Enron emails 1,600, because their pre-tokenised " , " text costs about 3
  characters per token); ≥ 4 sentences, ≥ 400 characters. No carrier may already contain a target string, "access granted", "secret key",
  "ignore all previous" or "<information>". Tensor Trust hijacks: 30–1,000 characters, contain "access granted", ≥ 30% distinct words.
- **Actual counts from v26a (seed 26):** carriers train 632 · dev 268 · test 389 · reserved 311. Rows 7,794, of which test attacks
  are 1,167. The README's "≈ 1,200 / 1,280" were approximations. Test attacks include about 190 Tensor Trust hijacks (G3).
- **Success rule details:**
  - normalisation strips leading quotes, `*`, `#` and `>`
  - G1 ignores S's trailing `.!?`
  - G2 is case- and separator-insensitive ("ab12cd34" matches "AB12-CD34")
  - unit-tested in v26b (I2)
- **Tokenisation:** prompts are encoded with `add_special_tokens=False` from the chat template string, with exactly one BOS for Llama
  (asserted). Earlier versions (v14b–v25) carried a double BOS on Llama-3-8B for consistency with v14b; v26 does not.
- **Last layer:** hooks store the residual stream **before** the final norm at every layer (HF's last `hidden_states` entry is after
  it); the fidelity check accounts for this.
- **V2:** 50 test attacks **per main model** (100 in total), stratified by goal (17 / 17 / 16), automatic label hidden from the rater.
- **D4 fallback:** if Hung et al.'s criterion (μ_n − μ_a − 4(σ_n + σ_a) > 0) selects fewer than 3 heads on the 60 selection rows,
  D4 uses the top 10 heads by (μ_n − μ_a)/(σ_n + σ_a). Which rule fired is saved.
- **D5:** score = p_exist × p_breach; L1-LR on z-scored per-head ratios, C ∈ {0.01, 0.1, 1} chosen on dev (ties → smaller C).
- **D2 / D3 / D6:** StandardScaler + LR (balanced), C ∈ {0.001, 0.01, 0.1, 1}; best dev AUROC, ties within 0.005 → layer nearest the
  middle, then the smaller C. Fitted on train only (not refit on train + dev).
- **I4 "position-only" baseline:** a single-turn prompt has no conversation position, so the baseline uses the prompt geometry (data start
  and end token, prompt length); "length-only" uses prompt and data token counts.
- **Check model (Llama-3-8B):** extracted on test + selection rows only, so only D4 (training-free) and raw success are reported for it.
- **v26d refuses to run** with fewer than 20 successful or 20 resisted training attacks (the breach detectors cannot be fitted).
- **v26e writes `_evaluated.lock`;** after that, v26d refuses to re-freeze.
- **Amendment to I5 (made before any run, found in the dry run):** "refit on random labels" is **not** guaranteed to give chance on
  test with hidden states. A few high-variance directions dominate, and a random-label probe still points along them with a random
  sign; the dry run gave AUROC up to 0.95 that way. I5 therefore follows the project's own convention (AGENTS.md: "feed the pipeline
  random arrays"): D2, D3 and D6 are refit with their selected C on **random Gaussian features of the same width, with the real
  labels**, and scored on the same random draw for test. AUROC must be within 0.5 ± 0.05, or the run stops.
- **v26c is two steps:** run Part 1 (it writes the sheet; Part 2 then stops with "not filled", which is expected), fill `followed`,
  save as CSV, re-run the notebook (Part 1 never overwrites a started sheet; Part 2 compares).
