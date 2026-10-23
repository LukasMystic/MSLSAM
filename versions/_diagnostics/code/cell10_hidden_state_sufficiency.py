# Snapshot of main.ipynb cell [10] -- extracted 2026-09-11, read-only archive.
# Do not edit here; edit main.ipynb and re-snapshot.

# =====================================================================
# HIDDEN STATE SUFFICIENCY TEST  --  MSLSAM
# Core question: is prompt injection reliably separable in hidden states?
#
# Motivation: multiple architectures (v8, v9, v10) converge to similar
# underperformance, suggesting a fundamental ceiling rather than an
# architecture problem.
#
# Experiments:
#   A. 4 classifiers × 2 training regimes -- does model family matter?
#   B. Dimensionality reduction -- do natural clusters exist?
#   C. Per-feature statistical separability -- Bhattacharyya, JSD, Fisher
#   D. DeepSet-only 5-fold CV -- best achievable under ideal conditions
#
# Decision criteria:
#   Exp D best AUC < 0.70 -> hidden states do not encode injection signal
#   Classifier AUC range  < 0.05 -> feature ceiling, not model ceiling
#   Max Fisher ratio < 0.5 -> features individually too weak
#
# Depends on: traj_clf_v9, monitor, system_prompt, attack_sources,
#   safe_sources, baseline_prompts, sensitive_safe_all, V9_INJECTION_CATEGORIES
# =====================================================================

try:
    from xgboost import XGBClassifier as _XGB
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import umap as _umap_mod
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA as _PCA2
from sklearn.manifold import TSNE
import copy as _cp
import matplotlib.pyplot as plt
import matplotlib.patches as _mp

# Sentinel values so final verdict compiles even if a section is skipped
_ds_aucs, _A_res = [], {}
_max_fisher = _mean_fisher = _max_bhatt = _max_jsd = float("nan")
_D_best_auc = float("nan")

print("\n" + "=" * 72)
print("HIDDEN STATE SUFFICIENCY TEST  --  MSLSAM")
print("  Is prompt injection reliably encoded in hidden states alone?")
print("=" * 72)
print(f"  XGBoost : {'available' if _HAS_XGB else 'not installed'}")
print(f"  UMAP    : {'available' if _HAS_UMAP else 'not installed'}")

# ─── Section 0: Feature Extraction (all cache hits) ──────────────────
print("\n[0] Extracting features (all cache hits expected)...")

def _ft(prompts, dname, desc=""):
    """Unscaled v9 26-dim feature matrix via cache."""
    raw = monitor.extract_states(system_prompt, prompts,
                                  dataset_name=dname, desc=desc)
    return traj_clf_v9._feature_matrix(raw)  # (N, 26)

_X_neg = np.vstack([
    _ft(baseline_prompts,   "baseline",       "SF-base"),
    _ft(sensitive_safe_all, "sensitive_safe", "SF-sens"),
])

_X_synt_parts = []
for _c, _e in V9_INJECTION_CATEGORIES.items():
    _cl = pp.preprocess_list(_e)
    _rw = monitor.extract_states(system_prompt, _cl,
                                  dataset_name=f"v9_cat_{_c.lower()}", desc=f"SF-{_c}")
    _X_synt_parts.append(traj_clf_v9._feature_matrix(_rw))
_X_synt = np.vstack(_X_synt_parts)

_pr_di = attack_sources.get("DeepSet-Injections", [])
_pr_db = safe_sources.get("DeepSet-Benign", [])
_pr_jb = safe_sources.get("JBB-Benign", [])

_X_di = (_ft(_pr_di, f"eval_deepset_injections_n{len(_pr_di)}", "SF-DSI")
          if _pr_di else np.empty((0, 26)))
_X_db = (_ft(_pr_db, f"eval_deepset_benign_n{len(_pr_db)}", "SF-DSB")
          if _pr_db else np.empty((0, 26)))
_X_jb = (_ft(_pr_jb, f"eval_jbb_benign_n{len(_pr_jb)}", "SF-JBB")
          if _pr_jb else np.empty((0, 26)))

print(f"  X_neg={len(_X_neg)}  X_synt={len(_X_synt)}"
      f"  X_ds_inj={len(_X_di)}  X_ds_ben={len(_X_db)}  X_jbb_ben={len(_X_jb)}")

# DS-Inj 80/20 split (consistent across experiments A and D)
np.random.seed(42)
_ndi = len(_X_di)
_pd  = np.random.permutation(_ndi)
_X_di_tr = _X_di[_pd[:int(0.8 * _ndi)]]
_X_di_ev = _X_di[_pd[int(0.8 * _ndi):]]


# ─── Experiment A: Classifier Comparison ─────────────────────────────
print("\n" + "─" * 72)
print("EXPERIMENT A: Classifier Comparison (4 families, 2 regimes)")
print("  A-Synt : Train synthetic + neg  -> Eval DS-Inj vs neg")
print("  A-DS   : Train DS-Inj(80%) + neg -> Eval DS-Inj(20%) vs neg")
print("─" * 72)

def _one_eval(clf, X_tr, y_tr, X_ne, X_ae):
    scl = RobustScaler().fit(X_tr)
    clf.fit(scl.transform(X_tr), y_tr)
    if len(X_ae) == 0:
        return float("nan"), float("nan"), float("nan")
    pr = clf.predict_proba(scl.transform(np.vstack([X_ne, X_ae])))[:, 1]
    sa, at = pr[:len(X_ne)].tolist(), pr[len(X_ne):].tolist()
    fc, tc, _ = roc_curve([0]*len(sa)+[1]*len(at), sa+at)
    _a = auc(fc, tc)
    _t = EvaluationEngine.find_optimal_threshold(sa, at)
    return (_a,
            sum(s >= _t for s in sa) / max(len(sa), 1) * 100,
            sum(s <  _t for s in at) / max(len(at), 1) * 100)

def _mk_suite():
    s = {
        "LR":  LogisticRegression(class_weight='balanced', max_iter=2000,
                                   solver='lbfgs', C=1.0, random_state=42, n_jobs=-1),
        "RF":  RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                       random_state=42, n_jobs=-1),
        "SVM": SVC(kernel='rbf', class_weight='balanced', probability=True,
                    C=1.0, gamma='scale', random_state=42),
    }
    if _HAS_XGB:
        s["XGB"] = _XGB(n_estimators=200, max_depth=4,
                         learning_rate=0.1, random_state=42, n_jobs=-1)
    return s

_Xts  = np.vstack([_X_neg, _X_synt])
_yts  = np.concatenate([np.zeros(len(_X_neg)), np.ones(len(_X_synt))])
_Xtd  = np.vstack([_X_neg, _X_di_tr])
_ytd  = np.concatenate([np.zeros(len(_X_neg)), np.ones(len(_X_di_tr))])

print(f"\n  {'Clf':<6}  {'A-Synt AUC':>11}  {'A-Synt FPR':>11}  "
      f"{'A-DS AUC':>9}  {'A-DS FPR':>9}  {'A-DS ASR':>9}")
print("  " + "-" * 64)

for _n, _clf in _mk_suite().items():
    _a1, _f1, _r1 = _one_eval(_cp.deepcopy(_clf), _Xts, _yts, _X_neg, _X_di)
    _a2, _f2, _r2 = _one_eval(_cp.deepcopy(_clf), _Xtd, _ytd, _X_neg, _X_di_ev)
    _A_res[_n] = {"synt_auc": _a1, "ds_auc": _a2, "ds_fpr": _f2, "ds_asr": _r2}
    def _q(v, pct=False):
        if np.isnan(v): return "   n/a" if pct else "    n/a"
        return f"{v:6.1f}%" if pct else f"{v:7.4f}"
    print(f"  {_n:<6}  {_q(_a1):>11}  {_q(_f1, True):>11}  "
          f"{_q(_a2):>9}  {_q(_f2, True):>9}  {_q(_r2, True):>9}")

_ds_aucs = [v["ds_auc"] for v in _A_res.values() if not np.isnan(v.get("ds_auc", float("nan")))]
if len(_ds_aucs) > 1:
    _rng = max(_ds_aucs) - min(_ds_aucs)
    print(f"\n  AUC range across classifiers (A-DS): {_rng:.4f}  "
          f"({'CONVERGED -- feature ceiling likely' if _rng < 0.05 else 'DIVERGED -- model complexity matters'})")
    print(f"  Best A-DS AUC: {max(_ds_aucs):.4f}  "
          f"({'below deployment threshold' if max(_ds_aucs) < 0.75 else 'possibly sufficient'})")


# ─── Experiment B: Dimensionality Reduction ──────────────────────────
print("\n" + "─" * 72)
print("EXPERIMENT B: Dimensionality Reduction & Visualization")
print("  Classes: DeepSet-Injection / DeepSet-Benign / JBB-Benign")
print("─" * 72)

_MAX_VIS = 400
np.random.seed(42)
def _sub(X, n):
    if len(X) == 0: return X
    idx = np.random.choice(len(X), min(n, len(X)), replace=False)
    return X[idx]

_Vi, _Vb, _Vj = _sub(_X_di, _MAX_VIS), _sub(_X_db, _MAX_VIS), _sub(_X_jb, _MAX_VIS)
_V_all   = [(v, n, c) for v, n, c in
             zip([_Vi, _Vb, _Vj],
                 ["DS-Injection", "DS-Benign", "JBB-Benign"],
                 ["crimson", "steelblue", "seagreen"])
             if len(v) > 0]

if len(_V_all) >= 2:
    _V_X   = np.vstack([v for v, _, _ in _V_all])
    _V_lbl = np.concatenate([np.full(len(v), i) for i, (v,_,_) in enumerate(_V_all)])
    _V_Xs  = RobustScaler().fit_transform(_V_X)

    _n_p = 2 + (1 if _HAS_UMAP else 0)
    fig, axes = plt.subplots(1, _n_p, figsize=(6*_n_p, 5))
    if _n_p == 1: axes = [axes]
    fig.suptitle("Latent Space Visualization: Injection vs Benign\n"
                 "(v9 26-dim features, RobustScaled, up to 400 samples/class)", fontsize=12)
    _patches = [_mp.Patch(color=c, label=n) for _, n, c in _V_all]

    # PCA 2D
    _pc2 = _PCA2(n_components=2, random_state=42).fit_transform(_V_Xs)
    for i, (_, _, c) in enumerate(_V_all):
        m = _V_lbl == i
        axes[0].scatter(_pc2[m, 0], _pc2[m, 1], c=c, s=12, alpha=0.5)
    axes[0].set_title("PCA (2D)"); axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].legend(handles=_patches, fontsize=8)
    print(f"  PCA 2D: {len(_V_Xs)} samples")

    # t-SNE 2D (limit to 600 total for speed)
    _ts_n = min(600, len(_V_Xs))
    _ts_idx = np.random.choice(len(_V_Xs), _ts_n, replace=False)
    _ts_X = _V_Xs[_ts_idx]; _ts_L = _V_lbl[_ts_idx]
    _perp = min(30, max(5, _ts_n // 10))
    print(f"  t-SNE 2D: {_ts_n} samples (perplexity={_perp})...")
    _tsne2 = TSNE(n_components=2, perplexity=_perp, random_state=42,
                   n_iter=800, verbose=0).fit_transform(_ts_X)
    for i, (_, _, c) in enumerate(_V_all):
        m = _ts_L == i
        axes[1].scatter(_tsne2[m, 0], _tsne2[m, 1], c=c, s=12, alpha=0.5)
    axes[1].set_title(f"t-SNE (2D, n={_ts_n})")
    axes[1].set_xlabel("Dim1"); axes[1].set_ylabel("Dim2")
    axes[1].legend(handles=_patches, fontsize=8)
    print(f"  t-SNE 2D: done")

    if _HAS_UMAP and len(axes) > 2:
        print(f"  UMAP 2D: {len(_V_Xs)} samples...")
        _um2 = _umap_mod.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                               random_state=42).fit_transform(_V_Xs)
        for i, (_, _, c) in enumerate(_V_all):
            m = _V_lbl == i
            axes[2].scatter(_um2[m, 0], _um2[m, 1], c=c, s=12, alpha=0.5)
        axes[2].set_title("UMAP (2D)")
        axes[2].set_xlabel("Dim1"); axes[2].set_ylabel("Dim2")
        axes[2].legend(handles=_patches, fontsize=8)
        print(f"  UMAP 2D: done")

    plt.tight_layout()
    plt.savefig("mslsam_sufficiency_dimred.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("[Saved: mslsam_sufficiency_dimred.png]")
else:
    print("  Need >= 2 classes for visualization -- check data availability")


# ─── Experiment C: Statistical Separability ──────────────────────────
print("\n" + "─" * 72)
print("EXPERIMENT C: Per-Feature Statistical Separability")
print("  DeepSet-Injection vs DeepSet-Benign  (all 26 v9 features)")
print("─" * 72)

def _bhatt(x1, x2):
    """Bhattacharyya distance (Gaussian approximation)."""
    m1, m2 = x1.mean(), x2.mean()
    s1, s2 = x1.std() + 1e-8, x2.std() + 1e-8
    return (0.25 * (m1-m2)**2 / (s1**2+s2**2) +
            0.5  * np.log(0.5*(s1**2/s2**2 + s2**2/s1**2 + 2)))

def _jsd(x1, x2, bins=60):
    """Jensen-Shannon divergence (histogram approximation)."""
    lo, hi = min(x1.min(), x2.min()), max(x1.max(), x2.max())
    if hi == lo: return 0.0
    dx = (hi - lo) / bins
    p, _ = np.histogram(x1, bins=bins, range=(lo, hi), density=True)
    q, _ = np.histogram(x2, bins=bins, range=(lo, hi), density=True)
    p += 1e-10; q += 1e-10
    m = 0.5 * (p + q)
    return float(0.5*(np.sum(p*np.log(p/m)) + np.sum(q*np.log(q/m))) * dx)

def _fisher(x1, x2):
    """Fisher discriminant ratio."""
    return float((x1.mean()-x2.mean())**2 / (x1.var()+x2.var()+1e-8))

if len(_X_di) >= 5 and len(_X_db) >= 5:
    _fnames = TrajectoryClassifierV9.FEATURE_NAMES
    _stats = []
    for fi in range(26):
        x1, x2 = _X_di[:, fi], _X_db[:, fi]
        _stats.append({
            "name": _fnames[fi],
            "bhatt": _bhatt(x1, x2),
            "jsd":   _jsd(x1, x2),
            "fisher": _fisher(x1, x2),
            "d": abs(x1.mean()-x2.mean()) / (np.sqrt((x1.var()+x2.var())/2)+1e-8),
        })
    _stats.sort(key=lambda s: -s["fisher"])

    print(f"\n  {'Feature':<25}  {'Bhatt':>6}  {'JSD':>8}  {'Fisher':>7}  {'Cohen_d':>8}")
    print("  " + "-" * 62)
    for st in _stats[:15]:
        print(f"  {st['name']:<25}  {st['bhatt']:>6.4f}"
              f"  {st['jsd']:>8.6f}  {st['fisher']:>7.4f}  {st['d']:>8.4f}")
    if len(_stats) > 15:
        print("  ...")

    _max_fisher  = max(s["fisher"] for s in _stats)
    _mean_fisher = float(np.mean([s["fisher"] for s in _stats]))
    _max_bhatt   = max(s["bhatt"]  for s in _stats)
    _max_jsd     = max(s["jsd"]    for s in _stats)
    _best_feat   = _stats[0]["name"]

    print(f"\n  Summary (all 26 features, DS-Inj vs DS-Ben):")
    print(f"    Fisher ratio  -- max={_max_fisher:.4f}  mean={_mean_fisher:.4f}"
          f"  ({'WEAK' if _max_fisher < 0.5 else 'MODERATE' if _max_fisher < 2.0 else 'STRONG'})")
    print(f"    Bhattacharyya -- max={_max_bhatt:.4f}")
    print(f"    JSD           -- max={_max_jsd:.6f}")
    print(f"    Best feature  : {_best_feat} (Fisher={_stats[0]['fisher']:.4f})")

    # Bar chart: Fisher ratio for top 14 features
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    _top = _stats[:14]
    axes[0].barh([s["name"] for s in _top][::-1],
                  [s["fisher"] for s in _top][::-1], color="steelblue")
    axes[0].axvline(1.0, color="red",    ls="--", lw=1, label="Fisher=1.0")
    axes[0].axvline(0.3, color="orange", ls="--", lw=1, label="Fisher=0.3")
    axes[0].set_xlabel("Fisher Discriminant Ratio")
    axes[0].set_title("Feature Separability (Fisher ratio)\nDS-Inj vs DS-Ben")
    axes[0].legend(fontsize=8)

    # Scatter: Fisher vs Bhattacharyya for all 26 features
    axes[1].scatter([s["fisher"] for s in _stats], [s["bhatt"] for s in _stats],
                     alpha=0.7, color="steelblue")
    for st in _stats[:6]:
        axes[1].annotate(st["name"].replace("P_","").replace("_L","\nL"),
                          (st["fisher"], st["bhatt"]), fontsize=6, alpha=0.8)
    axes[1].set_xlabel("Fisher Discriminant Ratio")
    axes[1].set_ylabel("Bhattacharyya Distance")
    axes[1].set_title("Feature Separability: Fisher vs Bhattacharyya\n(each dot = 1 feature)")

    plt.tight_layout()
    plt.savefig("mslsam_sufficiency_fisher.png", dpi=120, bbox_inches="tight")
    plt.show()
    print("[Saved: mslsam_sufficiency_fisher.png]")
else:
    print(f"  DeepSet-Injection (N={len(_X_di)}) or DeepSet-Benign (N={len(_X_db)})"
          f" too small -- skipped")


# ─── Experiment D: DeepSet-Only 5-Fold CV ────────────────────────────
print("\n" + "─" * 72)
print("EXPERIMENT D: DeepSet-Only 5-Fold Cross-Validation")
print("  Ideal distribution match: same source for train and test")
print("  Positive: DeepSet-Injections  |  Negative: DS-Benign (or base+sens)")
print("─" * 72)

if len(_X_di) >= 10:
    _X_neg_D = _X_db if len(_X_db) >= 10 else _X_neg
    _neg_src = "DS-Benign" if len(_X_db) >= 10 else "baseline+sensitive"
    print(f"\n  Negative pool: {_neg_src} (N={len(_X_neg_D)})")
    print(f"  Positive pool: DS-Inj (N={len(_X_di)})")

    _X_cv = np.vstack([_X_neg_D, _X_di])
    _y_cv = np.concatenate([np.zeros(len(_X_neg_D)), np.ones(len(_X_di))])
    _cv5  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print(f"\n  {'Classifier':<8}  {'Mean AUC':>10}  {'Std':>7}  "
          f"{'Min':>7}  {'Max':>7}")
    print("  " + "-" * 44)

    for _n, _clf in _mk_suite().items():
        pipe = Pipeline([("sc", RobustScaler()), ("clf", _cp.deepcopy(_clf))])
        cv_a = []
        for tr_i, te_i in _cv5.split(_X_cv, _y_cv):
            pipe.fit(_X_cv[tr_i], _y_cv[tr_i])
            pr = pipe.predict_proba(_X_cv[te_i])[:, 1]
            yt = _y_cv[te_i]
            if yt.sum() > 0 and (1-yt).sum() > 0:
                fc, tc, _ = roc_curve(yt, pr)
                cv_a.append(auc(fc, tc))
        if cv_a:
            _mu, _sd = np.mean(cv_a), np.std(cv_a)
            _D_best_auc = max(_D_best_auc if not np.isnan(_D_best_auc) else 0.0, _mu)
            print(f"  {_n:<8}  {_mu:>10.4f}  {_sd:>7.4f}  "
                  f"{min(cv_a):>7.4f}  {max(cv_a):>7.4f}")

    print(f"\n  Best mean CV AUC (ideal conditions): {_D_best_auc:.4f}")
    if _D_best_auc < 0.70:
        print("  -> AUC < 0.70 even under ideal matching. "
              "Hidden states likely insufficient.")
    elif _D_best_auc < 0.85:
        print("  -> AUC 0.70-0.85: weak but real signal. "
              "Calibration data is the bottleneck.")
    else:
        print("  -> AUC >= 0.85: strong signal. "
              "Architecture/calibration data can be improved.")
else:
    print(f"  DeepSet-Injections N={len(_X_di)} < 10 -- skipped")


# ─── Final Verdict ────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("SUFFICIENCY TEST: FINAL VERDICT")
print("=" * 72)

_bA  = max(_ds_aucs) if _ds_aucs else float("nan")
_rng = (max(_ds_aucs)-min(_ds_aucs)) if len(_ds_aucs) > 1 else float("nan")

print(f"""
Evidence:
  A. Best A-DS AUC (any classifier)  : {_bA:.4f if not np.isnan(_bA) else 'n/a'}
  A. Classifier AUC range (A-DS)     : {_rng:.4f if not np.isnan(_rng) else 'n/a'}  {'(CONVERGED)' if not np.isnan(_rng) and _rng < 0.05 else '(DIVERGED)' if not np.isnan(_rng) else ''}
  C. Max Fisher ratio (any feature)  : {_max_fisher:.4f if not np.isnan(_max_fisher) else 'n/a'}
  C. Mean Fisher ratio (all features): {_mean_fisher:.4f if not np.isnan(_mean_fisher) else 'n/a'}
  D. Best CV AUC (ideal conditions)  : {_D_best_auc:.4f if not np.isnan(_D_best_auc) else 'n/a'}
""")

_ceiling_reached = (not np.isnan(_D_best_auc) and _D_best_auc < 0.70)
_weak_signal     = (not np.isnan(_D_best_auc) and 0.70 <= _D_best_auc < 0.85)
_strong_signal   = (not np.isnan(_D_best_auc) and _D_best_auc >= 0.85)

if _ceiling_reached:
    print("""VERDICT: CEILING REACHED
  Hidden-state-only detection has reached its practical limit.

  The injection signal is NOT reliably encoded in early-layer hidden
  states at the granularity accessible via direction projections and
  trajectory features. Multiple classifiers confirm this: performance
  does not improve with model complexity.

  STOP: Do NOT invest further effort in:
    - Direction projections from new category sets
    - Trajectory or curvature features
    - Prototype-based similarity
    - Additional category consensus signals

  REDIRECT effort to fundamentally different signal sources:
    1. Instruction hierarchy tracking
       -- trace how instructions propagate through attention layers
    2. Attention flow analysis
       -- identify which tokens attend to injected content
    3. Context consistency checks
       -- does the model response contradict the system prompt?
    4. Output-space anomaly detection
       -- monitor response characteristics rather than input representations""")
elif _weak_signal:
    print("""VERDICT: WEAK SIGNAL -- CALIBRATION BOTTLENECK
  The feature space encodes a real but weak injection signal.
  The primary bottleneck is calibration data quality.

  RECOMMENDED next steps:
    1. Expand DeepSet-Injection into direction estimation
    2. Hard-negative expansion (confirmed effective in v10 design)
    3. Investigate attention-based features as supplementary signal
    4. Logit-space threshold or Platt scaling for score compression""")
elif _strong_signal:
    print("""VERDICT: SIGNAL PRESENT -- CALIBRATION DATA BOTTLENECK
  Hidden states encode sufficient injection signal.
  The primary bottleneck is calibration data, not feature space.

  RECOMMENDED next steps:
    1. Rebuild directions from real-world injection examples (DeepSet)
    2. Increase per-category calibration to N >= 50
    3. Hard-negative expansion already implemented in v10""")
else:
    print("""VERDICT: INCONCLUSIVE
  Insufficient evaluation data (DeepSet-Injections not available or too small).
  Obtain the DeepSet benchmark dataset to complete this investigation.""")

print("=" * 72)
