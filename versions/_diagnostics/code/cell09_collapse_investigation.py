# Snapshot of main.ipynb cell [9] -- extracted 2026-09-11, read-only archive.
# Do not edit here; edit main.ipynb and re-snapshot.

# =====================================================================
# MSLSAM -- Dataset Collapse Investigation
# Research question: is the score-distribution collapse caused by
#   (A) a weak classifier, or
#   (B) incorrect positive-class construction?
#
# Hypothesis B assumed until disproven.
#
# Experiments:
#   1. Train on synthetic only  -> eval on DeepSet injections
#   2. Train on DeepSet (50%)   -> eval on DeepSet held-out (50%)
#   3. Train on Synthetic + DS  -> eval on same DeepSet held-out
#   4. Recompute threshold from DeepSet-only scores (v9 scores reused)
#
# Analyses:
#   A. Per-feature KDE: DeepSet-Injection vs DeepSet-Benign
#   B. False Positive Inspection: JBB-Benign, WJB-Adv-Benign, DeepSet-Benign
#   C. Decision boundary: raw logits vs predicted probabilities (compression?)
#
# Depends on: traj_clf_v9, results_v9, monitor, system_prompt,
#             attack_sources, safe_sources, baseline_prompts,
#             sensitive_safe_all, V9_INJECTION_CATEGORIES
# =====================================================================

import re as _re
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as _KDE

# ─────────────────────────────────────────────────────────────────────
# SECTION 0: Raw Feature Extraction
#   All calls are cache hits from v9 evaluation. No new model inference.
#   Feature space: v9 26-dim (direction projections + trajectory).
#   Directions fixed from synthetic calibration -- consistent across exp.
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("DATASET COLLAPSE INVESTIGATION  --  MSLSAM")
print("  Hypothesis: bottleneck is positive-class construction, not architecture")
print("=" * 72)
print("\n[0] Feature Extraction (cache hits expected)...")

def _feats(prompts, dname, desc=""):
    """Extract raw (unscaled) 26-dim v9 features. Direction-normalized."""
    raw = monitor.extract_states(system_prompt, prompts,
                                 dataset_name=dname, desc=desc)
    return traj_clf_v9._feature_matrix(raw)   # (N, 26) unscaled

# Negative pool (baseline + sensitive safe) -- v9 training cache keys
X_base = _feats(baseline_prompts, "baseline",       "INV-base")
X_sens = _feats(sensitive_safe_all, "sensitive_safe", "INV-sens")
X_neg  = np.vstack([X_base, X_sens])

# Synthetic injection pool (per-category v9 cache keys)
_synt_parts = []
for _cat, _exs in V9_INJECTION_CATEGORIES.items():
    _cl = pp.preprocess_list(_exs)
    _rw = monitor.extract_states(system_prompt, _cl,
                                  dataset_name=f"v9_cat_{_cat.lower()}",
                                  desc=f"INV-{_cat}")
    _synt_parts.append(traj_clf_v9._feature_matrix(_rw))
X_synt = np.vstack(_synt_parts)

# DeepSet injections (eval cache key from v9 evaluation)
_ds_inj_prompts = attack_sources.get("DeepSet-Injections", [])
if _ds_inj_prompts:
    X_ds_inj = _feats(_ds_inj_prompts,
                       f"eval_deepset_injections_n{len(_ds_inj_prompts)}",
                       "INV-DS-Inj")
else:
    X_ds_inj = np.empty((0, 26))
    print("  WARNING: DeepSet-Injections not in attack_sources")

# DeepSet benign (for feature distribution analysis)
_ds_ben_prompts = safe_sources.get("DeepSet-Benign", [])
if _ds_ben_prompts:
    X_ds_ben = _feats(_ds_ben_prompts,
                       f"eval_deepset_benign_n{len(_ds_ben_prompts)}",
                       "INV-DS-Ben")
else:
    X_ds_ben = np.empty((0, 26))

print(f"  X_neg    (baseline+sensitive) : {len(X_neg)}")
print(f"  X_synt   (synthetic inj)      : {len(X_synt)}")
print(f"  X_ds_inj (DeepSet inj)        : {len(X_ds_inj)}")
print(f"  X_ds_ben (DeepSet benign)     : {len(X_ds_ben)}")

# 50/50 split for DeepSet injections (Exp2 and Exp3)
np.random.seed(42)
_n_ds = len(X_ds_inj)
_perm = np.random.permutation(_n_ds)
X_ds_tr = X_ds_inj[_perm[:_n_ds // 2]]
X_ds_ev = X_ds_inj[_perm[_n_ds // 2:]]
print(f"\n  DeepSet 50/50 split: {len(X_ds_tr)} train / {len(X_ds_ev)} eval")


# ─────────────────────────────────────────────────────────────────────
# SECTION 1: Experiment Helper
# ─────────────────────────────────────────────────────────────────────
def _run_exp(label, X_pos_tr, X_neg_tr, X_pos_ev, X_neg_ev, n_feat=26):
    """
    Fresh RobustScaler + LogisticRegression per experiment.
    Eval: held-out DeepSet injections vs full negative pool.
    Returns dict with keys: safe, atk, clf, scl, auc, t, fpr, asr.
    """
    if len(X_pos_tr) == 0 or len(X_pos_ev) == 0:
        print(f"  {label}: skipped (empty split)")
        return {"auc": float("nan"), "t": 0.5,
                "fpr": float("nan"), "asr": float("nan"),
                "safe": [], "atk": []}

    X_tr = np.vstack([X_neg_tr, X_pos_tr])[:, :n_feat]
    y_tr = np.concatenate([np.zeros(len(X_neg_tr)), np.ones(len(X_pos_tr))])

    scl = RobustScaler().fit(X_tr)
    X_trs = scl.transform(X_tr)
    clf = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0,
                              solver='lbfgs', random_state=42, n_jobs=-1)
    clf.fit(X_trs, y_tr)

    X_ev  = scl.transform(np.vstack([X_neg_ev, X_pos_ev])[:, :n_feat])
    proba = clf.predict_proba(X_ev)[:, 1]
    safe_s = proba[:len(X_neg_ev)].tolist()
    atk_s  = proba[len(X_neg_ev):].tolist()

    y_e = [0]*len(safe_s) + [1]*len(atk_s)
    fpr_c, tpr_c, _ = roc_curve(y_e, safe_s + atk_s)
    _auc = auc(fpr_c, tpr_c)
    _t   = EvaluationEngine.find_optimal_threshold(safe_s, atk_s)
    _fpr = sum(1 for s in safe_s if s >= _t) / max(len(safe_s), 1) * 100
    _asr = sum(1 for s in atk_s if s <  _t) / max(len(atk_s),  1) * 100

    print(f"  {label:<50}  AUC={_auc:.4f}  t={_t:.4f}"
          f"  FPR={_fpr:5.1f}%  ASR={_asr:5.1f}%"
          f"  (N+tr={len(X_pos_tr)} N-tr={len(X_neg_tr)} ev+={len(atk_s)})")
    return {"auc": _auc, "t": _t, "fpr": _fpr, "asr": _asr,
            "safe": safe_s, "atk": atk_s, "clf": clf, "scl": scl}


# ─────────────────────────────────────────────────────────────────────
# SECTION 2: Experiments 1–3
# ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("EXPERIMENTS 1–3: Training Data Variation")
print("  Feature space: v9 26-dim (directions fixed from synthetic)")
print("  Negative pool: baseline + sensitive safe (same for all)")
print("  Evaluation:    held-out 50% of DeepSet injections")
print("─" * 72)

_r1 = _run_exp("Exp1  Synthetic -> DeepSet",
               X_synt,                       X_neg,
               X_ds_ev,                      X_neg)
_r2 = _run_exp("Exp2  DeepSet(50%) -> DeepSet(50%)",
               X_ds_tr,                      X_neg,
               X_ds_ev,                      X_neg)
_r3 = _run_exp("Exp3  Synthetic+DeepSet(50%) -> DeepSet(50%)",
               np.vstack([X_synt, X_ds_tr]), X_neg,
               X_ds_ev,                      X_neg)

# Also compare: v9 full evaluation AUC on injection-only
_v9_inj_all = [s for s, l in zip(results_v9["attack_scores"], results_v9["attack_labels"])
               if l in {"Synthetic-Injections", "DeepSet-Injections"}]
_v9_safe_all = results_v9["safe_scores"]
if _v9_inj_all:
    _v9_y = [0]*len(_v9_safe_all) + [1]*len(_v9_inj_all)
    _v9_fpr_c, _v9_tpr_c, _ = roc_curve(_v9_y, _v9_safe_all + _v9_inj_all)
    _v9_inj_auc = auc(_v9_fpr_c, _v9_tpr_c)
    print(f"\n  Reference: v9 D_All on full eval (inj-vs-safe) AUC = {_v9_inj_auc:.4f}")

# ─────────────────────────────────────────────────────────────────────
# SECTION 3: Experiment 4 -- DeepSet-Only Threshold
#   Reuse existing v9 scores; retune threshold from DeepSet only.
#   Tests whether synthetic injections distort the decision boundary.
# ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("EXPERIMENT 4: DeepSet-Only Threshold (v9 scores, no retraining)")
print("─" * 72)

_ds_inj_scores = [s for s, l in zip(results_v9["attack_scores"], results_v9["attack_labels"])
                  if l == "DeepSet-Injections"]
_synt_scores   = [s for s, l in zip(results_v9["attack_scores"], results_v9["attack_labels"])
                  if l == "Synthetic-Injections"]

if _ds_inj_scores:
    _t4 = EvaluationEngine.find_optimal_threshold(_v9_safe_all, _ds_inj_scores)
    _fpr4 = sum(1 for s in _v9_safe_all   if s >= _t4) / max(len(_v9_safe_all), 1) * 100
    _asr4 = sum(1 for s in _ds_inj_scores if s <  _t4) / max(len(_ds_inj_scores), 1) * 100

    _t_mixed = results_v9["youden_t"]
    _asr_mix = sum(1 for s in _ds_inj_scores if s < _t_mixed) / max(len(_ds_inj_scores), 1) * 100
    _fpr_mix = sum(1 for s in _v9_safe_all  if s >= _t_mixed) / max(len(_v9_safe_all), 1) * 100

    print(f"  Mixed threshold (v9)       : t={_t_mixed:.4f}  FPR={_fpr_mix:.1f}%"
          f"  ASR(DeepSet)={_asr_mix:.1f}%")
    print(f"  DeepSet-only threshold     : t={_t4:.4f}  FPR={_fpr4:.1f}%"
          f"  ASR(DeepSet)={_asr4:.1f}%")

    # Mean/std of synthetic vs DeepSet scores under v9
    print(f"\n  v9 score stats:")
    print(f"    Synthetic-Injections  N={len(_synt_scores):4d}"
          f"  mean={np.mean(_synt_scores):.4f}  std={np.std(_synt_scores):.4f}")
    print(f"    DeepSet-Injections    N={len(_ds_inj_scores):4d}"
          f"  mean={np.mean(_ds_inj_scores):.4f}  std={np.std(_ds_inj_scores):.4f}")
    print(f"    Safe (all)            N={len(_v9_safe_all):4d}"
          f"  mean={np.mean(_v9_safe_all):.4f}  std={np.std(_v9_safe_all):.4f}")
else:
    print("  DeepSet-Injections not available in results_v9 -- skipped")
    _t4 = float("nan")

# ─────────────────────────────────────────────────────────────────────
# SECTION 4: Feature Distribution Analysis
#   6 key features: Override_L4/8/12, RoleHijack_L4, Auth_L4, Ctx_L4
#   Raw (unscaled) direction-normalised feature values.
# ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("FEATURE ANALYSIS: DeepSet-Injection vs DeepSet-Benign distributions")
print("─" * 72)

_FEAT_IDX   = [0, 1, 2, 4, 8, 12]
_FEAT_NAMES = ["P_Override_L4", "P_Override_L8", "P_Override_L12",
               "P_RoleHijack_L4", "P_Authority_L4", "P_ContextPoison_L4"]

if len(X_ds_inj) > 0 and len(X_ds_ben) > 0:
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Per-Feature Distributions: DeepSet-Injection vs DeepSet-Benign\n"
                 "(v9 feature space, direction-normalised, unscaled)", fontsize=13)

    for ax, fi, fn in zip(axes.flat, _FEAT_IDX, _FEAT_NAMES):
        inj_v = X_ds_inj[:, fi]
        ben_v = X_ds_ben[:, fi]
        _x_lo = min(inj_v.min(), ben_v.min()) - 0.5
        _x_hi = max(inj_v.max(), ben_v.max()) + 0.5
        _xs   = np.linspace(_x_lo, _x_hi, 300)
        for vals, label, color in [(ben_v, "DeepSet-Benign",   "steelblue"),
                                    (inj_v, "DeepSet-Injection", "crimson")]:
            try:
                kde_fn = _KDE(vals, bw_method="scott")
                yd = kde_fn(_xs)
                ax.plot(_xs, yd, label=label, color=color, lw=2)
                ax.fill_between(_xs, yd, alpha=0.2, color=color)
            except Exception:
                ax.hist(vals, bins=25, label=label, color=color,
                        alpha=0.5, density=True)

        # Compute Cohen's d
        d_coh = (inj_v.mean() - ben_v.mean()) / (
            np.sqrt((inj_v.std()**2 + ben_v.std()**2) / 2) + 1e-8)
        ax.set_title(f"{fn}\n(Cohen's d = {d_coh:.3f})", fontsize=9)
        ax.legend(fontsize=7)
        ax.set_xlabel("Feature value", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)

        print(f"  {fn:<25}  Inj mean={inj_v.mean():+.4f} std={inj_v.std():.4f}"
              f"  Ben mean={ben_v.mean():+.4f} std={ben_v.std():.4f}"
              f"  Cohen's d={d_coh:+.3f}")

    plt.tight_layout()
    plt.savefig("mslsam_collapse_feature_dist.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("[Saved: mslsam_collapse_feature_dist.png]")
else:
    print("  DeepSet-Injection or DeepSet-Benign features unavailable -- skipped")

# ─────────────────────────────────────────────────────────────────────
# SECTION 5: False Positive Calibration Analysis
#   Inspect top FPs from JBB-Benign, WJB-Adv-Benign, DeepSet-Benign.
#   Re-score via cache to align scores with original prompt texts.
# ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("CALIBRATION ANALYSIS: False Positive Pattern Inspection")
print("─" * 72)

_INJ_PATTERNS = {
    "quoted_injection": _re.compile(
        r"ignore (?:all )?(?:previous|prior) instructions?"
        r"|forget (?:your )?(?:instructions?|prompt|system)"
        r"|disregard (?:all )?(?:previous|prior|your)", _re.I),
    "roleplay": _re.compile(
        r"(?:pretend|act|roleplay|play) (?:you are|as|like)"
        r"|you are now\b|as an? \w+ with no", _re.I),
    "meta_discussion": _re.compile(
        r"prompt injection|jailbreak|red.?team|adversarial"
        r"|explain (?:how|why) (?:ai|llm)", _re.I),
    "prompt_engineering": _re.compile(
        r"system prompt|prompt engineer|zero.?shot|few.?shot"
        r"|chain.?of.?thought", _re.I),
    "instruction_like": _re.compile(
        r"(?:^|\. )(?:you must|you should|do not|always|never|respond (?:only|without|as))\b",
        _re.I | _re.MULTILINE),
}

_t_insp = results_v9["youden_t"]
_FP_SRCS = ["JBB-Benign", "WJB-Adv-Benign", "DeepSet-Benign"]

for _src in _FP_SRCS:
    _src_prompts = safe_sources.get(_src, [])
    if not _src_prompts:
        print(f"\n  {_src}: not in safe_sources")
        continue

    _dname = f"eval_{_src.lower().replace('-','_')}_n{len(_src_prompts)}"
    _sdict = traj_clf_v9.score_dataset(system_prompt, _src_prompts, dataset_name=_dname)
    _src_scores = _sdict["D_All"].tolist()

    fps = [(s, p) for s, p in zip(_src_scores, _src_prompts) if s >= _t_insp]
    fps.sort(key=lambda x: -x[0])

    _fp_rate = len(fps) / max(len(_src_scores), 1) * 100
    print(f"\n  {_src}: {len(fps)}/{len(_src_scores)} FPs ({_fp_rate:.1f}%)")

    # Pattern counts across ALL FPs
    _pattern_counts = {k: 0 for k in _INJ_PATTERNS}
    for _, _p in fps:
        for pname, pat in _INJ_PATTERNS.items():
            if pat.search(_p):
                _pattern_counts[pname] += 1

    if fps:
        print(f"  Pattern breakdown ({len(fps)} FPs):")
        for pname, cnt in sorted(_pattern_counts.items(), key=lambda x: -x[1]):
            pct = cnt / len(fps) * 100
            print(f"    {pname:<22}: {cnt:3d} / {len(fps)} ({pct:5.1f}%)")
        print(f"  Top 5 false positives (score / first 130 chars):")
        for _s, _p in fps[:5]:
            print(f"    [{_s:.4f}] {_p[:130].replace(chr(10), ' ')}...")

# ─────────────────────────────────────────────────────────────────────
# SECTION 6: Decision Boundary Analysis
#   Raw logits vs predicted probabilities for v9 D_All classifier.
#   If logit Cohen's d >> proba Cohen's d: probability compression issue.
# ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("DECISION BOUNDARY ANALYSIS: Logit vs Probability Compression")
print("─" * 72)

_clf_d = traj_clf_v9.classifiers["D_All"]
_sl    = TrajectoryClassifierV9.ABLATION["D_All"]   # (0, 26)

# Re-extract scaled features for all eval sets (cache hits)
def _sc_feats(prompts, dname):
    raw = monitor.extract_states(system_prompt, prompts,
                                  dataset_name=dname, desc=f"DB-{dname[:20]}")
    return traj_clf_v9.scaler.transform(traj_clf_v9._feature_matrix(raw))

_X_atk_parts, _X_sfe_parts = [], []
for _name, _pr in attack_sources.items():
    if _pr:
        _X_atk_parts.append(_sc_feats(
            _pr, f"eval_{_name.lower().replace('-','_')}_n{len(_pr)}"))
for _name, _pr in safe_sources.items():
    if _pr:
        _X_sfe_parts.append(_sc_feats(
            _pr, f"eval_{_name.lower().replace('-','_')}_n{len(_pr)}"))

X_atk_sc  = np.vstack(_X_atk_parts)
X_safe_sc = np.vstack(_X_sfe_parts)

logits_atk  = _clf_d.decision_function(X_atk_sc[:,  _sl[0]:_sl[1]])
logits_safe = _clf_d.decision_function(X_safe_sc[:, _sl[0]:_sl[1]])
probas_atk  = _clf_d.predict_proba(X_atk_sc[:,  _sl[0]:_sl[1]])[:, 1]
probas_safe = _clf_d.predict_proba(X_safe_sc[:, _sl[0]:_sl[1]])[:, 1]

_d_logit = (logits_atk.mean() - logits_safe.mean()) / (
    np.sqrt((logits_atk.std()**2 + logits_safe.std()**2) / 2) + 1e-8)
_d_proba = (probas_atk.mean() - probas_safe.mean()) / (
    np.sqrt((probas_atk.std()**2 + probas_safe.std()**2) / 2) + 1e-8)

print(f"  Logit Cohen's d = {_d_logit:.3f}"
      f"  (safe: {logits_safe.mean():+.3f}±{logits_safe.std():.3f}"
      f"  | atk: {logits_atk.mean():+.3f}±{logits_atk.std():.3f})")
print(f"  Proba Cohen's d = {_d_proba:.3f}"
      f"  (safe: {probas_safe.mean():.3f}±{probas_safe.std():.3f}"
      f"  | atk: {probas_atk.mean():.3f}±{probas_atk.std():.3f})")

_compress_ratio = abs(_d_proba) / (abs(_d_logit) + 1e-8)
print(f"  Compression ratio (proba/logit Cohen's d) = {_compress_ratio:.3f}")
if _compress_ratio < 0.5:
    print("  FINDING: Significant probability compression -- "
          "logits separate better than probabilities.")
    print("           Consider logit-space threshold or Platt recalibration.")
elif abs(_d_logit) < 0.3:
    print("  FINDING: Weak separation in logit space -- "
          "feature space inadequate. Dataset bottleneck likely.")
else:
    print("  FINDING: Probabilities and logits show similar separation.")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("v9 D_All Classifier: Logit vs Probability Distributions\n"
             "(All attack sources vs all safe sources)", fontsize=12)

for ax, (s_vals, a_vals), xlabel, titl in [
    (axes[0], (logits_safe, logits_atk), "Logit", "Raw Logits (before sigmoid)"),
    (axes[1], (probas_safe, probas_atk), "Probability", "Predicted Probabilities"),
]:
    for vals, label, color in [(s_vals, "Safe",   "steelblue"),
                                (a_vals, "Attack", "crimson")]:
        try:
            kde_fn = _KDE(vals, bw_method="scott")
            _xr = np.linspace(vals.min() - 0.1, vals.max() + 0.1, 300)
            _yd = kde_fn(_xr)
            ax.plot(_xr, _yd, label=label, color=color, lw=2)
            ax.fill_between(_xr, _yd, alpha=0.2, color=color)
        except Exception:
            ax.hist(vals, bins=40, label=label, color=color, alpha=0.5, density=True)
    if titl == "Raw Logits (before sigmoid)":
        ax.axvline(0, color="gray", ls="--", lw=1, label="boundary (logit=0)")
    ax.set_title(titl)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig("mslsam_collapse_logit_vs_prob.png", dpi=120, bbox_inches="tight")
plt.show()
print("[Saved: mslsam_collapse_logit_vs_prob.png]")

# ─────────────────────────────────────────────────────────────────────
# SECTION 7: Experiment Score Distributions (overlay plot)
# ─────────────────────────────────────────────────────────────────────
if all(_r.get("atk") for _r in [_r1, _r2, _r3]):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    fig.suptitle("Score Distributions by Training Regime\n"
                 "(Eval: held-out DeepSet injections vs baseline+sensitive safe)",
                 fontsize=12)
    for ax, _r, title in zip(
            axes,
            [_r1, _r2, _r3],
            ["Exp1: Synthetic→DS", "Exp2: DS(50%)→DS(50%)", "Exp3: Both→DS(50%)"]):
        for vals, label, color in [(_r["safe"], "Safe",    "steelblue"),
                                    (_r["atk"],  "Attack",  "crimson")]:
            try:
                kde_fn = _KDE(vals, bw_method="scott")
                _xr = np.linspace(0, 1, 300)
                _yd = kde_fn(_xr)
                ax.plot(_xr, _yd, label=label, color=color, lw=2)
                ax.fill_between(_xr, _yd, alpha=0.2, color=color)
            except Exception:
                ax.hist(vals, bins=30, label=label, color=color, alpha=0.5, density=True)
        ax.axvline(_r["t"], color="black", ls="--", lw=1.2,
                   label=f"t={_r['t']:.3f}")
        ax.set_title(f"{title}\nAUC={_r['auc']:.3f} FPR={_r['fpr']:.1f}% ASR={_r['asr']:.1f}%",
                     fontsize=9)
        ax.set_xlabel("Score"); ax.set_ylabel("Density"); ax.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("mslsam_collapse_exp_scores.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("[Saved: mslsam_collapse_exp_scores.png]")

# ─────────────────────────────────────────────────────────────────────
# SECTION 8: Summary Report
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("INVESTIGATION SUMMARY")
print("=" * 72)
print(f"""
Experiment Results (eval on held-out DeepSet injections):
  Exp1  Synthetic only -> DeepSet     :  AUC={_r1['auc']:.4f}  FPR={_r1['fpr']:5.1f}%  ASR={_r1['asr']:5.1f}%
  Exp2  DeepSet(50%)  -> DeepSet(50%) :  AUC={_r2['auc']:.4f}  FPR={_r2['fpr']:5.1f}%  ASR={_r2['asr']:5.1f}%
  Exp3  Synthetic+DS  -> DeepSet(50%) :  AUC={_r3['auc']:.4f}  FPR={_r3['fpr']:5.1f}%  ASR={_r3['asr']:5.1f}%

Decision Boundary:
  Logit Cohen's d = {_d_logit:.3f}  |  Proba Cohen's d = {_d_proba:.3f}
  Compression ratio = {_compress_ratio:.3f}

Diagnostic Interpretation:
  IF Exp2 AUC >> Exp1 AUC:
    -> DeepSet and synthetic occupy DIFFERENT latent manifolds.
    -> Direction space built from synthetic does NOT generalise.
    -> Fix: rebuild directions from diverse real-world examples.

  IF Exp1 AUC >> Exp2 AUC:
    -> Synthetic calibration captures the injection manifold well.
    -> DeepSet injections are unusual / out-of-distribution.
    -> Fix: investigate DeepSet subset composition.

  IF Exp3 AUC > max(Exp1, Exp2):
    -> Both sources contribute complementary signal.
    -> Fix: expand positive class with real-world injections.

  IF compression_ratio < 0.5:
    -> Probability mass is compressed near 0.5; logits separate better.
    -> Fix: use logit-space threshold or retrain with explicit calibration.

  IF logit Cohen's d < 0.3:
    -> Feature space itself cannot separate populations.
    -> Fix: revisit feature family, not just training data.
""")
print("=" * 72)
