# v21 recap — what was done, what went right, what went wrong (written 2026-10-11, before any v21 verdict exists)

## 1. The question v21 was built to answer
v20b's headline: MSLSAM-Δ beats Prompt Guard 2 on the length-clean indirect families (BIPIA + InjecAgent), pooled ASR 63.9 vs 78.2, in 5 of 5 seeds.
That result was pre-registered, but every version from v17 to v20b had looked at the same four test families. The design that produced it
(tool-output training data, the Δ feature, the choice of comparator) was therefore shaped by the test set.

v21 asked two things:
1. **Does the frozen v20b detector keep its result on data no version has seen?**
2. **How does the defence run in deployment, and what does it cost?** This is the inference mode plus time and space complexity, for MSLSAM
   alone and for the MSLSAM + PG2 ensemble.

## 2. What was built (2026-09-29)
- **The pre-registration was written first** (`versions/20_mslsam-v21/README.md`): integrity check I1, C1 (MSLSAM-Δ vs PG2 on fresh data,
  BEATS/MATCHES/LOSES, ≥ 4/5 seeds), C2 (does the z-mean ensemble earn its cost?), the target, a length flag (AUC > 0.60), and descriptive views.
- **Fresh test data (1,800 rows):** 600 XSum BBC articles, chosen by BIPIA's own summarization index. The task is "Summarize the following
  news article in one sentence." Each article has three length-matched versions: injected, clean (declarative filler) and hard negative
  (a Dolly instruction). Payload positions are start, middle or end. The data is capped at 1,800 characters so it fits PG2's 512-token window.
  - **Sum-BIPIA:** BIPIA's 150 text attacks. While building, I found that **all of BIPIA's attack strings were already in the v19 test set**,
    so this family has new carriers and a new task but old payload strings.
  - **Sum-TT:** 150 Tensor Trust human-written hijacks (GitHub), never used in the project. Added because of the finding above.
- **Freezing the detector:** v20b is refitted per seed, and the refit must reproduce v20b's saved test scores. The fitted pieces are saved as
  `artifacts/mslsam_v20b_s*.npz` with a 15-line numpy scorer (`mslsam_head.py`). This is literally what a deployment would load.
- **Ensemble:** z-mean, (z_Δ + z_PG2)/2, with μ/σ from external + tool benign rows only (out-of-fold for MSLSAM). Chosen over v20a's z-max
  because a max is hijacked by whichever score has drifted. With per-family thresholds, μ drops out and the mean is a fixed weighting of the two logits.
- **Notebooks:** v21a (freeze + build, CPU) → v21b (Llama extraction + latency benchmark + judge + guards, GPU) → v21c (verdicts + cost table).
  All three were dry-run end to end on mocks and synthetic states.

## 3. What happened, in order
| when | run | outcome |
|---|---|---|
| 09-29 16:13–16:22 | **v21a** | ✅ **OK.** XSum md5s match BIPIA's (exactly BIPIA's articles). Tensor Trust: 194 usable, 150 drawn, none contaminated. Length-only AUC 0.500 / 0.506 (length-clean). Max 729 tokens. **I1 part 1 passed:** refit = v20b to 3e-08, numpy scorer = sklearn to 3.7e-06. OOF AUC 0.997 |
| 09-29 16:2x → 09-30 02:35 | **v21b run 1** | ⚠ Extraction **completed and saved** (8 shards, fresh arrays). Shards 0–2 took ~1 h each; after a restart, shards 3–7 took ~4 min each. Then the kernel **died** with no Python error, in the benchmark or guard-loading cell. Nothing from the benchmark was written |
| 09-30 | fix 1 | Llama and the guards were split into separate kernels (v21b + v21b2), and the benchmark saved after every prompt |
| 09-30 | **v21b run 2** | ❌ Sat for 2 h at "B1: 0 prompts already timed, 50 to do" with **no output** (inside the warm-up, which printed nothing) |
| 09-30 | fix 2 | Added a pre-flight (memory/swap report, one prompt with every stage printed), per-prompt progress and a heartbeat file |
| 09-30 → 10-10 | **v21b run 3** | ❌ The extraction speed gate **failed** (0.20 rows/s, below the limit). **Swap 50.1 GB used of 51.2 GB.** Prefill took 2.5 s for 282 tokens (healthy is ~0.3 s). Then it **hung** in "decode 4 tokens" (heartbeat 12:56:37) |
| 10-10 | decision | **The Llama latency benchmark is abandoned.** v21b2 became standalone (guards only), and v21c cites the Llama costs from v17's healthy run |

**Status now:**
- v21a is done.
- v21b's extraction is done; its benchmark is abandoned.
- **v21b2 and v21c have not been run, so no C1/C2 verdict exists yet.**

## 4. What went right
1. **Pre-registration before numbers.** Even after three failed runs, nothing has been computed on the fresh families, so the confirmation is still clean.
2. **Integrity was proven, not assumed.** The frozen model is v20b's (3e-08), the deployable numpy scorer is sklearn's (4e-06), and the articles are BIPIA's (md5).
3. **Catching the payload reuse.** Checking BIPIA's attack strings against the v19 test set before the run showed they were not fresh. Tensor Trust gave a truly unseen payload family.
4. **Length-matching worked.** Length alone cannot separate the fresh families (AUC 0.50). This is unlike SPML, where length alone gets 0.994.
5. **Resumable shards saved the expensive part.** Every crash and hang came after the 1,800-row extraction had been saved.
6. **The heartbeat file and memory report turned a mystery into a diagnosis.** Run 2 hung silently; run 3 named the exact stage and showed the swap.

## 5. What went wrong — and my part in it
1. **One GPU notebook did too much.** I coupled the confirmation (which needs only hidden states) with a latency benchmark (which needs a healthy,
   quiet machine) and with three extra models. The result was that one failure mode blocked everything. *Lesson: separate what answers the
   research question from what measures cost.*
2. **I co-loaded Llama-8B fp16 (≈ 16 GB) with two fp32 guards (≈ 1.9 GB) on a 24 GB Mac** without checking headroom. That is the likely cause of
   run 1's crash. Earlier versions always ran the guards in their own kernel; I broke that pattern.
3. **I missed the warning signs.** Run 1's first shards were ~15× slower than the later ones. That was swap, and I noted it only after the crash.
   *Lesson: a slow run is a symptom, not just an inconvenience.*
4. **No progress output inside long loops.** Run 2's warm-up printed nothing for 2 hours, so we could not tell "slow" from "stuck". *Lesson:
   every long loop prints per item, and a pre-flight times one item first.*
5. **The pre-flight checked the wrong thing first.** Run 3's pre-flight printed 50 GB of swap but only planned to stop *after* the stages finished,
   and the decode never finished. It should have refused to start when swap was high. Also, the extraction smoke gate had already failed
   (0.20 rows/s). *Lesson: a failed gate means stop. Never force past it, and gates should check the machine state up front.*
6. **Dry runs gave false confidence.** Mocks prove that the code runs. They cannot reveal memory pressure, MPS behaviour or real speed.
7. **Unresolved:** whether the decode hang was purely swap or an MPS/KV-cache problem. Prefill finished (slowly) but decode never did. This was not
   isolated, and it should be stated as unknown.

## 6. What must be disclosed in the thesis (deviations from the pre-registration)
- **I1 part 3 was not run.** That check was "the deployment path (batch 1, KV cache kept) gives the same score as the batched extraction".
  Parts 1–2 passed.
- **The 8B judge was not run on the fresh rows.**
- **These were not measured; theory only:** hook overhead on the Apple GPU, decoding from the kept KV cache, and how Llama prefill time grows with length.
- **The Llama prefill (326 ms) and judge (378.5 ms) costs are v17's** batch-1 measurements on a healthy run, cited. The MSLSAM scorer cost is
  timed in v21c on the frozen artifact (CPU). PG2's latency comes from v21b2.
- **Sum-BIPIA reuses payload strings from the v19 test set.** Sum-TT is the fully unseen family.
- **The fresh families get one look only.** Any later version needs new fresh data.

## 7. Concepts used (for learning)
- **Pre-registration:** fix the rules before seeing results, so the result cannot shape the rules.
- **Freeze + reproduce:** proving the deployed file *is* the evaluated model (max |Δ| 3e-08) closes a common gap between paper and system.
- **Out-of-fold scores:** a model scores its own training rows too optimistically. Each training row is therefore scored by a model that never saw it,
  and those scores give honest reference statistics.
- **z-mean vs z-max:** a max lets one drifted detector dominate. A mean weights both, by 1/σ.
- **Per-family thresholds vs deployment thresholds:** the verdict calibrates on each family's clean rows. The deployment view asks what happens
  without that step.
- **Swap:** when RAM is full, macOS writes memory to disk. A model whose 16 GB of weights are partly on disk must page them back for every step.
  Decoding, which touches all the weights for every single token, is hit hardest.
- **KV cache:** keeping the attention keys and values from the prefill means the answer can be generated without a second pass. That is the
  basis of "MSLSAM costs ~0 extra Llama passes".
- **Complexity:** Llama prefill is O(L·(n·d² + n²·d)) and is paid by the app anyway. The MSLSAM scorer is O(K·d·k) ≈ 8.4 M multiply-adds,
  constant in prompt length. PG2 is a separate model, linear in the number of 512-token windows. The judge is a second full prefill.

## 8. Before planning
- Restart the Mac (to clear the swap) → run v21b2_guards → run v21c → evaluate C1/C2 → then plan.
