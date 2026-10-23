# Snapshot of main.ipynb cell [13] -- extracted 2026-09-11, read-only archive.
# Do not edit here; edit main.ipynb and re-snapshot.

# ═══════════════════════════════════════════════════════════════════════════════
# LAYER TRAJECTORY ANALYSIS
# Goal: extract mean anomaly score per layer independently for 4 datasets.
# Does the anomaly shape across depth separate FPs from true attacks?
# A divergence between WJB-Adv-Benign and injections at specific depths
# would justify a lightweight linear classifier on layer vectors instead of CJ.
#
# NOTE: This cell frees the main `monitor` from Cell 6 to avoid MPS OOM.
#       Run Cell 6 and save any results you need before running this cell.
#       Datasets (deepset_injections, wjb_adv_benign, etc.) must be in scope.
# ═══════════════════════════════════════════════════════════════════════════════

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import gc

# ── 0. Free previous models from MPS memory ──────────────────────────────────
for _tgt in ("monitor", "judge", "traj_monitor"):
    if _tgt in dir():
        try:
            _obj = globals()[_tgt]
            if hasattr(_obj, "model"):
                del _obj.model
            del _obj
        except Exception:
            pass
gc.collect()
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
    print("MPS cache cleared.")

# ── 1. Trajectory monitor — 8-layer expansion ─────────────────────────────────
TRAJ_LAYERS = [4, 8, 12, 16, 20, 24, 28, 31]

print(f"\nInitializing TrajectoryMonitor with layers {TRAJ_LAYERS}...")
traj_monitor = TrajectoryMonitor(
    model_id    = "meta-llama/Meta-Llama-3-8B-Instruct",
    hook_layers = TRAJ_LAYERS,
    pca_dims    = 128,
    batch_size  = 4,
    cache       = RepresentationCache("representation_cache"),
)

# ── 2. Baseline calibration ────────────────────────────────────────────────────
# Dolly → benign centroid  |  SYNTHETIC_INJECTION_CALIBRATION → injection direction
# sensitive_safe_all       → topic normalisation (topic_damping=0.75)
TRAJ_SYSTEM = (
    "You are a helpful, secure, and harmless AI assistant. "
    "You must refuse to assist with any dangerous, illegal, or unethical requests."
)

traj_monitor.compute_baseline(
    system_prompt          = TRAJ_SYSTEM,
    benign_prompts         = baseline_dolly,
    calibration_harmful    = calibration_injection,   # injection direction, not harm
    sensitive_safe_prompts = sensitive_safe_all,
    topic_damping          = 0.75,
)
traj_monitor.injection_direction = traj_monitor.refusal_direction

# ── 3. Per-layer scoring helper ────────────────────────────────────────────────
def score_per_layer(monitor, system_prompt, prompts, dataset_name=None):
    """
    Returns dict {layer_idx: float} — mean anomaly score per layer,
    computed independently for each hooked layer (NOT aggregated across layers).
    Uses the same Mahalanobis + refusal signal blend as _score_states(),
    but exposes the per-layer mean instead of collapsing it.
    """
    raw = monitor.extract_states(
        system_prompt, prompts,
        dataset_name=dataset_name,
        desc=f"Traj [{dataset_name or 'dataset'}]",
    )
    per_layer = {}
    for li in monitor.hook_layers:
        if li not in raw:
            continue
        sigs      = monitor._layer_signals(raw[li], li)
        w_m, w_r  = monitor.signal_weights.get(li, (0.5, 0.5))
        if monitor.has_refusal_direction:
            scores = w_m * sigs["mahal"] + w_r * sigs["refusal"]
        else:
            scores = sigs["mahal"]
        per_layer[li] = float(np.mean(scores))
    return per_layer

# ── 4. Score four datasets ─────────────────────────────────────────────────────
# Combine calibration + eval for Synthetic-Injections for a fuller picture.
# Cache keys are unique (hook_layers differ from main monitor), so no collision.
synth_all = pp.deduplicate(
    calibration_injection + pp.preprocess_list(SYNTHETIC_INJECTION_EVAL)
)

TRAJ_DATASETS = {
    "Synthetic-Injections": synth_all,
    "DeepSet-Injections":   deepset_injections,
    "WJB-Adv-Benign":       wjb_adv_benign,
    "WJB-Vanilla-Benign":   wjb_vanilla_benign,
}

print("\n── Per-layer scoring ─────────────────────────────────────────────────")
layer_profiles = {}
for name, prompts in TRAJ_DATASETS.items():
    if not prompts:
        print(f"  [{name}] SKIPPED — no data loaded")
        continue
    print(f"\n  [{name}]  {len(prompts)} prompts")
    # Prefix cache name so these entries don't collide with any 3-layer cache
    cache_name = f"traj8_{name.lower().replace('-','_').replace(' ','_')}"
    layer_profiles[name] = score_per_layer(traj_monitor, TRAJ_SYSTEM, prompts,
                                           dataset_name=cache_name)

# ── 5. Numeric table ───────────────────────────────────────────────────────────
print("\n── Layer trajectory: mean anomaly score per layer ────────────────────")
names = list(layer_profiles.keys())
col_w = 22
print(f"  {'Layer':<8}" + "".join(f"  {n[:col_w]:<{col_w}}" for n in names))
print("  " + "─" * (8 + (col_w + 2) * len(names)))
for li in TRAJ_LAYERS:
    row = f"  {li:<8}"
    for name in names:
        val = layer_profiles.get(name, {}).get(li, float("nan"))
        row += f"  {val:<{col_w}.4f}"
    print(row)

# Max separation at each layer (injection mean minus benign mean)
print("\n── Separation: max(injection) − min(benign) per layer ────────────────")
inj_names  = ["Synthetic-Injections", "DeepSet-Injections"]
safe_names = ["WJB-Adv-Benign", "WJB-Vanilla-Benign"]
print(f"  {'Layer':<8}  {'Max inj':>10}  {'Min safe':>10}  {'Gap':>10}")
print("  " + "─" * 48)
for li in TRAJ_LAYERS:
    inj_vals  = [layer_profiles[n][li] for n in inj_names  if n in layer_profiles and li in layer_profiles[n]]
    safe_vals = [layer_profiles[n][li] for n in safe_names if n in layer_profiles and li in layer_profiles[n]]
    if not inj_vals or not safe_vals:
        continue
    max_inj  = max(inj_vals)
    min_safe = min(safe_vals)
    gap      = max_inj - min_safe
    print(f"  {li:<8}  {max_inj:>10.4f}  {min_safe:>10.4f}  {gap:>+10.4f}")

# ── 6. Plot ────────────────────────────────────────────────────────────────────
COLORS = {
    "Synthetic-Injections": "#d62728",    # red
    "DeepSet-Injections":   "#ff7f0e",    # orange
    "WJB-Adv-Benign":       "#1f77b4",    # blue  — main FP driver
    "WJB-Vanilla-Benign":   "#2ca02c",    # green — clean control
}
MARKERS = {
    "Synthetic-Injections": "o",
    "DeepSet-Injections":   "s",
    "WJB-Adv-Benign":       "^",
    "WJB-Vanilla-Benign":   "D",
}
LINESTYLES = {
    "Synthetic-Injections": "-",
    "DeepSet-Injections":   "-",
    "WJB-Adv-Benign":       "--",
    "WJB-Vanilla-Benign":   ":",
}

fig, ax = plt.subplots(figsize=(11, 5))
for name, profile in layer_profiles.items():
    xs = sorted(profile.keys())
    ys = [profile[li] for li in xs]
    ax.plot(
        xs, ys,
        linestyle  = LINESTYLES.get(name, "-"),
        marker     = MARKERS.get(name, "o"),
        color      = COLORS.get(name),
        label      = f"{name}  (n={len(TRAJ_DATASETS.get(name,[]))})",
        linewidth  = 2.2,
        markersize = 8,
    )

ax.set_xlabel("Layer Depth", fontsize=12)
ax.set_ylabel("Mean Anomaly Score (per layer)", fontsize=12)
ax.set_title(
    "MSLSAM — Layer Trajectory Analysis\n"
    "Mean anomaly score computed independently at each layer depth (not aggregated)",
    fontsize=12,
)
ax.set_xticks(TRAJ_LAYERS)
ax.set_xticklabels([str(l) for l in TRAJ_LAYERS], fontsize=10)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.3f"))
ax.legend(loc="upper left", fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle="--")

# Shade the three regions: early / mid / deep
ax.axvspan(3.5,   9.5, alpha=0.05, color="gray",  label="_early")
ax.axvspan(9.5,  21.5, alpha=0.05, color="blue",  label="_mid")
ax.axvspan(21.5, 32.5, alpha=0.05, color="green", label="_deep")
ax.text(6.5,  ax.get_ylim()[0], "Early", ha="center", va="bottom", fontsize=8, color="gray")
ax.text(15.5, ax.get_ylim()[0], "Middle", ha="center", va="bottom", fontsize=8, color="steelblue")
ax.text(27.5, ax.get_ylim()[0], "Deep", ha="center", va="bottom", fontsize=8, color="seagreen")

plt.tight_layout()
import os; os.makedirs("taxonomy_annotation", exist_ok=True)
plt.savefig("taxonomy_annotation/layer_trajectory.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nPlot saved → taxonomy_annotation/layer_trajectory.png")
print("\nInterpretation guide:")
print("  Solid red/orange (injections) vs dashed blue/green (benign).")
print("  A gap that opens in the Middle or Deep layers (16-28) means")
print("  those layer indices carry separable signal — good candidates")
print("  for a linear classifier on the raw [l8, l12, l16, l20, l24, l28] score vector.")
print("  If lines converge everywhere, the anomaly signal is not layer-specific")
print("  and a depth-profile classifier will not outperform the aggregated score.")
