# Snapshot of main.ipynb cell [11] -- extracted 2026-09-11, read-only archive.
# Do not edit here; edit main.ipynb and re-snapshot.

# =====================================================================
# MSLSAM v8 REPRESENTATION AUDIT
# Core question: does the v8 latent representation contain sufficient
# information for real-world injection detection, independent of the
# downstream classifier?
#
# v9 and v10 share the same monitor (same hidden states).
# If the representation is the bottleneck, no downstream architecture
# can rescue performance. This audit tests that directly.
#
# Experiments:
#   1. Model ceiling    -- 4 classifiers on raw v8 features vs DeepSet
#   2. DS-only CV       -- 5-fold, ideal distribution match
#   3. Feature audit    -- Fisher, Bhattacharyya, JSD per v8 feature
#   4. Visualization    -- PCA / t-SNE / UMAP of v8 feature space
#   5. Threshold collapse -- logits vs proba; locate collapse origin
#   6. Synthetic contamination -- are synthetic prompts off-manifold?
#
# v8 feature vector: 16-dim (8 Mahalanobis + 8 Refusal-Direction)
#   one value per monitored layer [4,8,12,16,20,24,28,31]
#   Mahal  = L2 norm of (pca - baseline_mu) / sqrt(eigenvalues)
#   RefDir = (pca - baseline_mu) @ monitor.injection_direction[layer]
#
# Depends on: monitor, traj_clf (v8), traj_clf_v9, system_prompt,
#   attack_sources, safe_sources, baseline_prompts, sensitive_safe_all
# =====================================================================

try:
    from xgboost import XGBClassifier as _XGB;  _HAS_XGB  = True
except ImportError:
    _HAS_XGB = False

try:
    import umap as _umap_mod; _HAS_UMAP = True
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

# Global sentinels (populated per experiment)
_E2_best_auc = float("nan")
_E3_max_fisher = _E3_mean_fisher = float("nan")
_E5_d_logit = _E5_d_proba = _E5_compress = float("nan")

print("\n" + "=" * 72)
print("MSLSAM v8 REPRESENTATION AUDIT")
print("  Does the v8 latent representation contain sufficient information")
print("  for real-world injection detection, independent of classifier?")
print("=" * 72)
print(f"  XGBoost : {'available' if _HAS_XGB else 'not installed'}")
print(f"  UMAP    : {'available' if _HAS_UMAP else 'not installed'}")

# ─────────────────────────────────────────────────────────────────────
# SECTION 0: v8 Feature Extraction
# ─────────────────────────────────────────────────────────────────────
print("\n[0] Building v8 feature extraction functions...")

# Resolve injection direction dict  ──────────────────────────────────
try:
    _raw_inj_dir = monitor.injection_direction
    _inj_dirs = (_raw_inj_dir if isinstance(_raw_inj_dir, dict)
                 else {li: _raw_inj_dir for li in monitor.hook_layers})
    print(f"  monitor.injection_direction: dict with {len(_inj_dirs)} layers")
except AttributeError:
    _inj_dirs = {}
    print("  [warn] monitor.injection_direction not found -- RefDir will be zero")

def _v8_feats(raw_states):
    """
    Compute v8 16-dim feature vector.
    First 8: Mahalanobis distance in PCA space (per monitored layer).
    Last 8:  Refusal-direction projection    (per monitored layer).
    Falls back to traj_clf._feature_matrix if available.
    """
    # Try to use v8 classifier's own feature_matrix first
    try:
        return traj_clf._feature_matrix(raw_states)
    except (AttributeError, TypeError, KeyError):
        pass

    layers = [li for li in monitor.hook_layers if li in raw_states]
    mahal_cols, ref_cols = [], []
    for li in layers:
        pca   = monitor.baseline_pca[li].transform(raw_states[li])
        delta = pca - monitor.baseline_mu[li]
        # Mahalanobis in PCA space: normalise by sqrt(eigenvalue)
        ev    = monitor.baseline_pca[li].explained_variance_ + 1e-8
        mahal = np.sqrt(np.sum(delta ** 2 / ev, axis=1))
        mahal_cols.append(mahal)
        # Refusal direction projection
        d = _inj_dirs.get(li, np.zeros(pca.shape[1]))
        ref_cols.append(delta @ d)

    # Pad to 8 columns if fewer layers present
    _zero = np.zeros(len(mahal_cols[0])) if mahal_cols else np.zeros(0)
    while len(mahal_cols) < 8: mahal_cols.append(_zero.copy())
    while len(ref_cols)   < 8: ref_cols.append(_zero.copy())
    return np.column_stack(mahal_cols[:8] + ref_cols[:8])  # (N, 16)

def _extract(prompts, dname, desc="", feats_fn=None):
    """Extract hidden states (cached) → feature matrix."""
    raw = monitor.extract_states(system_prompt, prompts,
                                  dataset_name=dname, desc=desc)
    return feats_fn(raw) if feats_fn else raw

print("[0] Extracting features (all cache hits expected)...")

# Standard safe pools
_X_v8_neg = np.vstack([
    _v8_feats(monitor.extract_states(system_prompt, baseline_prompts,
              dataset_name="baseline", desc="AU-base")),
    _v8_feats(monitor.extract_states(system_prompt, sensitive_safe_all,
              dataset_name="sensitive_safe", desc="AU-sens")),
])

# Synthetic injection pool
_X_v8_synt_p = []
for _c, _e in V9_INJECTION_CATEGORIES.items():
    _cl = pp.preprocess_list(_e)
    _rw = monitor.extract_states(system_prompt, _cl,
                                  dataset_name=f"v9_cat_{_c.lower()}", desc=f"AU-{_c}")
    _X_v8_synt_p.append(_v8_feats(_rw))
_X_v8_synt = np.vstack(_X_v8_synt_p)

# DeepSet injection and benign
_pr_di = attack_sources.get("DeepSet-Injections", [])
_pr_db = safe_sources.get("DeepSet-Benign", [])
_pr_jb = safe_sources.get("JBB-Benign", [])

def _ds(prompts, dname, desc):
    if not prompts: return np.empty((0, 16))
    return _v8_feats(monitor.extract_states(system_prompt, prompts,
                                             dataset_name=dname, desc=desc))

_X_v8_di = _ds(_pr_di, f"eval_deepset_injections_n{len(_pr_di)}",  "AU-DSI")
_X_v8_db = _ds(_pr_db, f"eval_deepset_benign_n{len(_pr_db)}",      "AU-DSB")
_X_v8_jb = _ds(_pr_jb, f"eval_jbb_benign_n{len(_pr_jb)}",          "AU-JBB")

_v8_dim = _X_v8_di.shape[1] if len(_X_v8_di) else 16
print(f"  v8 feature dim : {_v8_dim}")
print(f"  X_neg={len(_X_v8_neg)}  X_synt={len(_X_v8_synt)}"
      f"  X_di={len(_X_v8_di)}  X_db={len(_X_v8_db)}  X_jb={len(_X_v8_jb)}")

# 80/20 split for DeepSet (reproducible)
np.random.seed(42)
_pdi = np.random.permutation(len(_X_v8_di))
_X_di_tr = _X_v8_di[_pdi[:int(0.8*len(_pdi))]]
_X_di_ev = _X_v8_di[_pdi[int(0.8*len(_pdi)):]]

# Classifier suite factory
def _suite():
    s = {"LR":  LogisticRegression(class_weight='balanced', max_iter=2000,
                                    solver='lbfgs', C=1.0, random_state=42, n_jobs=-1),
         "RF":  RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                        random_state=42, n_jobs=-1),
         "SVM": SVC(kernel='rbf', class_weight='balanced', probability=True,
                     C=1.0, gamma='scale', random_state=42)}
    if _HAS_XGB:
        s["XGB"] = _XGB(n_estimators=200, max_depth=4, learning_rate=0.1,
                         random_state=42, n_jobs=-1)
    return s

def _one(clf, X_tr, y_tr, X_ne, X_ae):
    if len(X_ae) == 0: return float("nan"), float("nan"), float("nan")
    scl = RobustScaler().fit(X_tr)
    clf.fit(scl.transform(X_tr), y_tr)
    pr   = clf.predict_proba(scl.transform(np.vstack([X_ne, X_ae])))[:, 1]
    sa, at = pr[:len(X_ne)].tolist(), pr[len(X_ne):].tolist()
    fc, tc, _ = roc_curve([0]*len(sa)+[1]*len(at), sa+at)
    _a = auc(fc, tc)
    _t = EvaluationEngine.find_optimal_threshold(sa, at)
    return (_a,
            sum(s >= _t for s in sa) / max(len(sa), 1) * 100,
            sum(s <  _t for s in at) / max(len(at), 1) * 100)

# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 1: Model Ceiling (v8 features, 4 classifiers)
# ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("EXPERIMENT 1: Model Ceiling Analysis")
print("  Feature space: v8 raw features (NOT through MoE RF)")
print("  A-Synt: Train synthetic + neg -> Eval DS-Inj vs neg")
print("  A-DS  : Train DS-Inj(80%) + neg -> Eval DS-Inj(20%) vs neg")
print("─" * 72)

_Xts = np.vstack([_X_v8_neg, _X_v8_synt])
_yts = np.concatenate([np.zeros(len(_X_v8_neg)), np.ones(len(_X_v8_synt))])
_Xtd = np.vstack([_X_v8_neg, _X_di_tr])
_ytd = np.concatenate([np.zeros(len(_X_v8_neg)), np.ones(len(_X_di_tr))])

_E1_results, _E1_ds_aucs = {}, []
print(f"\n  {'Clf':<6}  {'A-Synt AUC':>11}  {'A-Synt FPR':>11}  "
      f"{'A-DS AUC':>9}  {'A-DS FPR':>9}  {'A-DS ASR':>9}")
print("  " + "-" * 64)

for _n, _clf in _suite().items():
    _a1, _f1, _r1 = _one(_cp.deepcopy(_clf), _Xts, _yts, _X_v8_neg, _X_v8_di)
    _a2, _f2, _r2 = _one(_cp.deepcopy(_clf), _Xtd, _ytd, _X_v8_neg, _X_di_ev)
    _E1_results[_n] = (_a1, _f1, _r1, _a2, _f2, _r2)
    if not np.isnan(_a2): _E1_ds_aucs.append(_a2)
    def _q(v, p=False):
        return (f"{v:6.1f}%" if p else f"{v:7.4f}") if not np.isnan(v) else ("  n/a" if p else "   n/a")
    print(f"  {_n:<6}  {_q(_a1):>11}  {_q(_f1,True):>11}  "
          f"{_q(_a2):>9}  {_q(_f2,True):>9}  {_q(_r2,True):>9}")

if len(_E1_ds_aucs) > 1:
    _rng = max(_E1_ds_aucs) - min(_E1_ds_aucs)
    print(f"\n  Classifier AUC range (A-DS): {_rng:.4f}  "
          f"({'CONVERGED -> representation ceiling' if _rng < 0.05 else 'DIVERGED -> model complexity matters'})")
    print(f"  Best A-DS AUC: {max(_E1_ds_aucs):.4f}")

# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 2: DeepSet-Only 5-Fold Cross-Validation
# ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("EXPERIMENT 2: DeepSet-Only 5-Fold CV (leak-free ceiling estimate)")
print("  Positive: DS-Inj  |  Negative: DS-Ben (or base+sens if unavailable)")
print("─" * 72)

if len(_X_v8_di) >= 10:
    _X_neg_D = _X_v8_db if len(_X_v8_db) >= 10 else _X_v8_neg
    _src_D   = "DS-Benign" if len(_X_v8_db) >= 10 else "baseline+sensitive"
    _X_cv  = np.vstack([_X_neg_D, _X_v8_di])
    _y_cv  = np.concatenate([np.zeros(len(_X_neg_D)), np.ones(len(_X_v8_di))])
    _cv5   = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    print(f"\n  Neg: {len(_X_neg_D)} ({_src_D})  |  Pos: {len(_X_v8_di)} (DS-Inj)")
    print(f"\n  {'Clf':<6}  {'Mean AUC':>10}  {'Std':>7}  {'Min':>7}  {'Max':>7}")
    print("  " + "-" * 44)

    _E2_best_auc = 0.0
    for _n, _clf in _suite().items():
        _pipe = Pipeline([("sc", RobustScaler()), ("clf", _cp.deepcopy(_clf))])
        _cv_a = []
        for _tri, _tei in _cv5.split(_X_cv, _y_cv):
            _pipe.fit(_X_cv[_tri], _y_cv[_tri])
            _pr = _pipe.predict_proba(_X_cv[_tei])[:, 1]
            _yt = _y_cv[_tei]
            if _yt.sum() > 0 and (1-_yt).sum() > 0:
                _fc, _tc, _ = roc_curve(_yt, _pr)
                _cv_a.append(auc(_fc, _tc))
        if _cv_a:
            _mu, _sd = np.mean(_cv_a), np.std(_cv_a)
            _E2_best_auc = max(_E2_best_auc, _mu)
            print(f"  {_n:<6}  {_mu:>10.4f}  {_sd:>7.4f}  "
                  f"{min(_cv_a):>7.4f}  {max(_cv_a):>7.4f}")

    print(f"\n  Best mean CV AUC: {_E2_best_auc:.4f}  -> "
          f"{'INSUFFICIENT (<0.75)' if _E2_best_auc < 0.75 else 'WEAK (0.75-0.85)' if _E2_best_auc < 0.85 else 'STRONG (>=0.85)'}")
else:
    print(f"  DS-Inj N={len(_X_v8_di)} -- insufficient for CV")

# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 3: Per-Feature Separability Audit
# ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("EXPERIMENT 3: Per-Feature Separability (DS-Inj vs DS-Ben)")
print("─" * 72)

def _bhatt(x1, x2):
    m1,m2 = x1.mean(), x2.mean(); s1,s2 = x1.std()+1e-8, x2.std()+1e-8
    return 0.25*(m1-m2)**2/(s1**2+s2**2) + 0.5*np.log(0.5*(s1**2/s2**2+s2**2/s1**2+2))

def _jsd(x1, x2, bins=60):
    lo,hi = min(x1.min(),x2.min()), max(x1.max(),x2.max())
    if hi==lo: return 0.0
    dx = (hi-lo)/bins
    p,_ = np.histogram(x1, bins=bins, range=(lo,hi), density=True)
    q,_ = np.histogram(x2, bins=bins, range=(lo,hi), density=True)
    p+=1e-10; q+=1e-10; m=0.5*(p+q)
    return float(0.5*(np.sum(p*np.log(p/m))+np.sum(q*np.log(q/m)))*dx)

def _fdr(x1, x2):
    return float((x1.mean()-x2.mean())**2/(x1.var()+x2.var()+1e-8))

def _cohend(x1, x2):
    return abs(x1.mean()-x2.mean())/(np.sqrt((x1.var()+x2.var())/2)+1e-8)

_E3_stats = []
if len(_X_v8_di) >= 5 and len(_X_v8_db) >= 5:
    # Feature names for v8
    _hook = [li for li in monitor.hook_layers][:8]
    _v8_names = ([f"Mahal_L{li}" for li in _hook] +
                 [f"RefDir_L{li}" for li in _hook])

    for fi in range(min(_v8_dim, len(_v8_names))):
        x1, x2 = _X_v8_di[:, fi], _X_v8_db[:, fi]
        _E3_stats.append({"name": _v8_names[fi],
                           "fisher": _fdr(x1, x2), "bhatt": _bhatt(x1, x2),
                           "jsd": _jsd(x1, x2), "d": _cohend(x1, x2)})
    _E3_stats.sort(key=lambda s: -s["fisher"])
    _E3_max_fisher  = max(s["fisher"] for s in _E3_stats)
    _E3_mean_fisher = float(np.mean([s["fisher"] for s in _E3_stats]))

    print(f"\n  {'Feature':<16}  {'Fisher':>8}  {'Bhatt':>7}  {'JSD':>9}  {'Cohen_d':>8}")
    print("  " + "-" * 54)
    for st in _E3_stats:
        print(f"  {st['name']:<16}  {st['fisher']:>8.4f}  {st['bhatt']:>7.4f}"
              f"  {st['jsd']:>9.6f}  {st['d']:>8.4f}")

    print(f"\n  Summary (v8 {_v8_dim} features, DS-Inj vs DS-Ben):")
    print(f"    Fisher max={_E3_max_fisher:.4f}  mean={_E3_mean_fisher:.4f}"
          f"  -> {'WEAK' if _E3_max_fisher < 0.5 else 'MODERATE' if _E3_max_fisher < 2.0 else 'STRONG'}")
    print(f"    Best feature: {_E3_stats[0]['name']} (Fisher={_E3_stats[0]['fisher']:.4f})")
    print(f"    Worst: {_E3_stats[-1]['name']} (Fisher={_E3_stats[-1]['fisher']:.4f})")

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh([s["name"] for s in _E3_stats][::-1],
                  [s["fisher"] for s in _E3_stats][::-1], color="steelblue")
    axes[0].axvline(1.0, color="red",    ls="--", lw=1, label="Fisher=1.0")
    axes[0].axvline(0.3, color="orange", ls="--", lw=1, label="Fisher=0.3")
    axes[0].set_xlabel("Fisher Discriminant Ratio"); axes[0].set_title("Feature Separability (all v8 features)")
    axes[0].legend(fontsize=8)

    axes[1].scatter([s["fisher"] for s in _E3_stats],
                     [s["bhatt"]  for s in _E3_stats], alpha=0.8, color="steelblue")
    for st in _E3_stats[:4]:
        axes[1].annotate(st["name"], (st["fisher"], st["bhatt"]), fontsize=7, alpha=0.85)
    axes[1].set_xlabel("Fisher Discriminant Ratio"); axes[1].set_ylabel("Bhattacharyya Dist")
    axes[1].set_title("Fisher vs Bhattacharyya (v8 features, DS-Inj vs DS-Ben)")
    plt.tight_layout()
    plt.savefig("mslsam_audit_exp3_separability.png", dpi=120, bbox_inches="tight")
    plt.show(); print("[Saved: mslsam_audit_exp3_separability.png]")
else:
    print(f"  DS-Inj N={len(_X_v8_di)} or DS-Ben N={len(_X_v8_db)} too small")

# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 4: Latent Visualization
# ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("EXPERIMENT 4: Latent Visualization of v8 Feature Space")
print("  Groups: DS-Inj (red), DS-Ben (blue), JBB-Ben (green), Synt-Inj (orange)")
print("─" * 72)

_MAX_V = 400
np.random.seed(42)
def _sub(X, n):
    if len(X)==0: return X
    return X[np.random.choice(len(X), min(n, len(X)), replace=False)]

_Vdi = _sub(_X_v8_di, _MAX_V); _Vdb = _sub(_X_v8_db, _MAX_V)
_Vjb = _sub(_X_v8_jb, _MAX_V); _Vsy = _sub(_X_v8_synt, _MAX_V)
_V_all = [(v,n,c) for v,n,c in [(_Vdi,"DS-Inj","crimson"),(_Vdb,"DS-Ben","steelblue"),
                                  (_Vjb,"JBB-Ben","seagreen"),(_Vsy,"Synt-Inj","orange")]
          if len(v)>0]

if len(_V_all) >= 2:
    _VX = np.vstack([v for v,_,_ in _V_all])
    _VL = np.concatenate([np.full(len(v), i) for i,(v,_,_) in enumerate(_V_all)])
    _VXs = RobustScaler().fit_transform(_VX)
    _np = 2 + (1 if _HAS_UMAP else 0)
    fig, axes = plt.subplots(1, _np, figsize=(6*_np, 5))
    if _np == 1: axes = [axes]
    fig.suptitle("v8 Feature Space: 4 Groups\n(RobustScaled, up to 400/class)", fontsize=12)
    _patches = [_mp.Patch(color=c, label=n) for _,n,c in _V_all]

    _pc2 = _PCA2(n_components=2, random_state=42).fit_transform(_VXs)
    for i,(_,_,c) in enumerate(_V_all):
        m = _VL==i; axes[0].scatter(_pc2[m,0], _pc2[m,1], c=c, s=15, alpha=0.5)
    axes[0].set_title("PCA 2D"); axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    axes[0].legend(handles=_patches, fontsize=8)

    _tn = min(800, len(_VXs)); _tidx = np.random.choice(len(_VXs), _tn, replace=False)
    _tX = _VXs[_tidx]; _tL = _VL[_tidx]; _perp = min(30, max(5, _tn//10))
    print(f"  t-SNE: {_tn} samples (perplexity={_perp})...")
    _ts2 = TSNE(n_components=2, perplexity=_perp, random_state=42,
                 n_iter=800, verbose=0).fit_transform(_tX)
    for i,(_,_,c) in enumerate(_V_all):
        m = _tL==i; axes[1].scatter(_ts2[m,0], _ts2[m,1], c=c, s=15, alpha=0.5)
    axes[1].set_title(f"t-SNE 2D (n={_tn})")
    axes[1].set_xlabel("Dim1"); axes[1].set_ylabel("Dim2")
    axes[1].legend(handles=_patches, fontsize=8)

    if _HAS_UMAP and len(axes)>2:
        print("  UMAP...")
        _um2 = _umap_mod.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                                random_state=42).fit_transform(_VXs)
        for i,(_,_,c) in enumerate(_V_all):
            m=_VL==i; axes[2].scatter(_um2[m,0], _um2[m,1], c=c, s=15, alpha=0.5)
        axes[2].set_title("UMAP 2D")
        axes[2].set_xlabel("Dim1"); axes[2].set_ylabel("Dim2")
        axes[2].legend(handles=_patches, fontsize=8)

    plt.tight_layout()
    plt.savefig("mslsam_audit_exp4_visualization.png", dpi=120, bbox_inches="tight")
    plt.show(); print("[Saved: mslsam_audit_exp4_visualization.png]")

# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 5: Threshold Collapse Investigation
# ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("EXPERIMENT 5: Threshold Collapse — Locating the Origin")
print("  Compare v9 D_All raw logits vs predicted probabilities.")
print("  If logit Cohen's d >> proba Cohen's d: probability compression.")
print("  If logit Cohen's d ≈ 0: feature space itself is flat.")
print("─" * 72)

try:
    _clf_d = traj_clf_v9.classifiers["D_All"]
    _s9, _e9 = TrajectoryClassifierV9.ABLATION["D_All"]   # (0, 26)

    # Scaled v9 features for all attack and safe sources (cache hits)
    def _sc9(prompts, dname):
        raw = monitor.extract_states(system_prompt, prompts, dataset_name=dname,
                                      desc=f"E5-{dname[:15]}")
        return traj_clf_v9.scaler.transform(traj_clf_v9._feature_matrix(raw))

    _X9_atk = np.vstack([_sc9(_pr, f"eval_{_n.lower().replace('-','_')}_n{len(_pr)}")
                           for _n, _pr in attack_sources.items() if _pr])
    _X9_sfe = np.vstack([_sc9(_pr, f"eval_{_n.lower().replace('-','_')}_n{len(_pr)}")
                           for _n, _pr in safe_sources.items() if _pr])

    _logit_atk  = _clf_d.decision_function(_X9_atk[:, _s9:_e9])
    _logit_safe = _clf_d.decision_function(_X9_sfe[:, _s9:_e9])
    _proba_atk  = _clf_d.predict_proba(_X9_atk[:, _s9:_e9])[:, 1]
    _proba_safe = _clf_d.predict_proba(_X9_sfe[:, _s9:_e9])[:, 1]

    def _cohd(a, b): return (a.mean()-b.mean())/(np.sqrt((a.var()+b.var())/2)+1e-8)
    _E5_d_logit  = _cohd(_logit_atk, _logit_safe)
    _E5_d_proba  = _cohd(_proba_atk, _proba_safe)
    _E5_compress = abs(_E5_d_proba)/(abs(_E5_d_logit)+1e-8)

    print(f"\n  Logit Cohen's d = {_E5_d_logit:+.4f}"
          f"  (atk={_logit_atk.mean():+.3f}±{_logit_atk.std():.3f}"
          f"  safe={_logit_safe.mean():+.3f}±{_logit_safe.std():.3f})")
    print(f"  Proba Cohen's d = {_E5_d_proba:+.4f}"
          f"  (atk={_proba_atk.mean():.3f}±{_proba_atk.std():.3f}"
          f"  safe={_proba_safe.mean():.3f}±{_proba_safe.std():.3f})")
    print(f"  Compression ratio (|proba_d|/|logit_d|) = {_E5_compress:.4f}")

    if _E5_compress < 0.4 and abs(_E5_d_logit) > 0.3:
        print("\n  FINDING: PROBABILITY COMPRESSION")
        print("  Logits separate classes better than probabilities.")
        print("  Sigmoid is compressing a real signal into near-0.5 space.")
        print("  Logit-space threshold or Platt recalibration would help.")
    elif abs(_E5_d_logit) < 0.2:
        print("\n  FINDING: FEATURE FLATNESS")
        print("  Even raw logits show poor separation.")
        print("  The feature space itself does not encode injection signal.")
    else:
        print("\n  FINDING: MODERATE SIGNAL")
        print("  Some compression present; logit threshold worth testing.")

    # Plot logit and proba distributions
    from scipy.stats import gaussian_kde as _KDE2
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Threshold Collapse: v9 D_All Classifier\n"
                 "Raw Logits vs Predicted Probabilities", fontsize=12)
    for ax, (sv, av), xl, tl in [
        (axes[0], (_logit_safe, _logit_atk), "Logit", "Raw Logits"),
        (axes[1], (_proba_safe, _proba_atk), "Probability", "Predicted Probabilities"),
    ]:
        for vals, lbl, col in [(sv,"Safe","steelblue"),(av,"Attack","crimson")]:
            try:
                kde = _KDE2(vals, bw_method="scott")
                xr = np.linspace(vals.min()-0.1, vals.max()+0.1, 300)
                yd = kde(xr); ax.plot(xr, yd, color=col, lw=2, label=lbl)
                ax.fill_between(xr, yd, alpha=0.2, color=col)
            except Exception:
                ax.hist(vals, bins=40, color=col, alpha=0.5, density=True, label=lbl)
        if xl == "Logit": ax.axvline(0, color="gray", ls="--", lw=1, label="boundary")
        ax.set_title(tl); ax.set_xlabel(xl); ax.set_ylabel("Density"); ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("mslsam_audit_exp5_collapse.png", dpi=120, bbox_inches="tight")
    plt.show(); print("[Saved: mslsam_audit_exp5_collapse.png]")

except Exception as _e5_err:
    print(f"  [skip] Experiment 5 error: {_e5_err}")

# ─────────────────────────────────────────────────────────────────────
# EXPERIMENT 6: Synthetic Contamination Audit
# ─────────────────────────────────────────────────────────────────────
print("\n" + "─" * 72)
print("EXPERIMENT 6: Synthetic Contamination Audit")
print("  Are synthetic injections on the same manifold as real injections?")
print("  Key test: d(Synt, DS-Ben) vs d(Synt, DS-Inj).")
print("  If Synt closer to DS-Ben than DS-Inj -> synthetic is off-manifold.")
print("─" * 72)

if len(_X_v8_synt) > 0 and len(_X_v8_di) > 0:
    # Fit shared scaler on all data
    _scl6 = RobustScaler().fit(np.vstack([_X_v8_neg, _X_v8_synt,
                                            _X_v8_di, _X_v8_db] if len(_X_v8_db) else
                                           [_X_v8_neg, _X_v8_synt, _X_v8_di]))
    _Xs_sy = _scl6.transform(_X_v8_synt)
    _Xs_di = _scl6.transform(_X_v8_di)
    _Xs_db = _scl6.transform(_X_v8_db) if len(_X_v8_db) else None
    _Xs_ng = _scl6.transform(_X_v8_neg)

    _c_sy = _Xs_sy.mean(axis=0)
    _c_di = _Xs_di.mean(axis=0)
    _c_ng = _Xs_ng.mean(axis=0)
    _c_db = _Xs_db.mean(axis=0) if _Xs_db is not None else None

    _d_sy_di = np.linalg.norm(_c_sy - _c_di)
    _d_sy_ng = np.linalg.norm(_c_sy - _c_ng)
    _d_di_ng = np.linalg.norm(_c_di - _c_ng)

    print(f"\n  Centroid distances (scaled v8 space, L2):")
    print(f"    Synt-Inj -> DS-Inj  : {_d_sy_di:.4f}")
    print(f"    Synt-Inj -> Neg     : {_d_sy_ng:.4f}")
    print(f"    DS-Inj   -> Neg     : {_d_di_ng:.4f}")
    if _c_db is not None:
        _d_sy_db = np.linalg.norm(_c_sy - _c_db)
        _d_di_db = np.linalg.norm(_c_di - _c_db)
        print(f"    Synt-Inj -> DS-Ben  : {_d_sy_db:.4f}  *** KEY METRIC ***")
        print(f"    DS-Inj   -> DS-Ben  : {_d_di_db:.4f}")
        if _d_sy_db < _d_sy_di:
            print(f"\n  FINDING: SYNTHETIC CONTAMINATION CONFIRMED")
            print(f"  Synthetic injections are CLOSER to DS-Benign ({_d_sy_db:.4f})")
            print(f"  than to DS-Injection ({_d_sy_di:.4f}).")
            print(f"  Detector is learning synthetic prompt artifacts, not injection behavior.")
        else:
            print(f"\n  FINDING: Synthetic injections occupy a different region than benign.")
            print(f"  Synthetic-to-real injection distance may still indicate distribution gap.")

    # Per-feature Cohen's d: Synt-Inj vs DS-Inj
    print(f"\n  Per-feature Cohen's d (Synt-Inj vs DS-Inj):")
    _hook_names = ([f"Mahal_L{li}" for li in [li for li in monitor.hook_layers][:8]] +
                   [f"RefDir_L{li}" for li in [li for li in monitor.hook_layers][:8]])
    _sy_vs_di = []
    for fi in range(min(_v8_dim, len(_hook_names))):
        _sy_vs_di.append((_hook_names[fi],
                            _cohend(_X_v8_synt[:, fi], _X_v8_di[:, fi])))
    _sy_vs_di.sort(key=lambda x: -x[1])
    for nm, dv in _sy_vs_di:
        bar = "█" * int(dv * 10)
        print(f"    {nm:<16} {dv:6.3f}  {bar}")

    # 2D scatter: all 4 groups in first 2 PCA dims
    _all6 = [(v,n,c) for v,n,c in
              [(_Xs_sy,"Synt-Inj","orange"),(_Xs_di,"DS-Inj","crimson"),
               (_Xs_ng,"Safe-Neg","lightblue")]
              + ([(_Xs_db,"DS-Ben","steelblue")] if _Xs_db is not None else [])
              if len(v)>0]
    _VX6 = np.vstack([v for v,_,_ in _all6])
    _VL6 = np.concatenate([np.full(len(v), i) for i,(v,_,_) in enumerate(_all6)])
    _pc6 = _PCA2(n_components=2, random_state=42).fit_transform(_VX6)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i,(_,n,c) in enumerate(_all6):
        m = _VL6==i
        ax.scatter(_pc6[m,0], _pc6[m,1], c=c, s=15, alpha=0.5, label=n)
    ax.set_title("Synthetic Contamination: PCA of v8 Space\n"
                 "All 4 groups (up to 400/class)")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("mslsam_audit_exp6_contamination.png", dpi=120, bbox_inches="tight")
    plt.show(); print("[Saved: mslsam_audit_exp6_contamination.png]")

else:
    print("  Insufficient data for contamination audit")

# ─────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("REPRESENTATION AUDIT: FINAL REPORT")
print("=" * 72)

_best_E1 = max(_E1_ds_aucs) if _E1_ds_aucs else float("nan")
_rng_E1  = (max(_E1_ds_aucs)-min(_E1_ds_aucs)) if len(_E1_ds_aucs)>1 else float("nan")

def _nf(v): return f"{v:.4f}" if not np.isnan(v) else "n/a"

print(f"""
Experiment Results:
  E1 Best A-DS AUC (any classifier)   : {_nf(_best_E1)}
  E1 Classifier AUC range             : {_nf(_rng_E1)}  ({'CONVERGED' if not np.isnan(_rng_E1) and _rng_E1 < 0.05 else 'DIVERGED' if not np.isnan(_rng_E1) else 'n/a'})
  E2 Best CV AUC (DeepSet-only)       : {_nf(_E2_best_auc)}
  E3 Max Fisher ratio (v8 features)   : {_nf(_E3_max_fisher)}
  E3 Mean Fisher ratio (v8 features)  : {_nf(_E3_mean_fisher)}
  E5 Logit Cohen's d                  : {_nf(_E5_d_logit)}
  E5 Proba Cohen's d                  : {_nf(_E5_d_proba)}
  E5 Compression ratio                : {_nf(_E5_compress)}
""")

# Verdict logic
_is_repr_sufficient = not np.isnan(_E2_best_auc) and _E2_best_auc >= 0.75
_is_ceiling          = not np.isnan(_E2_best_auc) and _E2_best_auc < 0.75
_is_compressed       = not np.isnan(_E5_compress) and _E5_compress < 0.4 and abs(_E5_d_logit) > 0.3
_is_feature_flat     = not np.isnan(_E5_d_logit)  and abs(_E5_d_logit) < 0.2

print("─" * 72)
print("ANSWERS TO RESEARCH QUESTIONS")
print("─" * 72)
print(f"""
Q1. Best achievable DeepSet-only AUC with v8 representation:
    {_nf(_E2_best_auc)}  (Experiment 2, ideal distribution matching)

Q2. Can the v8 representation realistically achieve FPR<10%, ASR<5%?
    {'UNLIKELY' if _is_ceiling else 'POSSIBLE BUT CONSTRAINED' if not np.isnan(_E2_best_auc) and _E2_best_auc < 0.85 else 'PLAUSIBLE' if not np.isnan(_E2_best_auc) else 'UNKNOWN (insufficient DS data)'}
    An AUC < 0.75 cannot support both constraints simultaneously at any
    threshold. AUC > 0.85 is required for operational deployment.

Q3. Primary bottleneck:
    {'[PROBABILITY COMPRESSION] Signal exists in logit space but sigmoid squashes it. Try logit-space threshold.' if _is_compressed else
     '[FEATURE FLATNESS] Raw logits already show poor separation. Features do not encode injection.' if _is_feature_flat else
     '[REPRESENTATION QUALITY] Some signal present but weak. Calibration data is likely compounding the issue.' if _is_repr_sufficient else
     '[REPRESENTATION CEILING] The v8 16-dim feature space appears to have reached its information ceiling for real-world injections.'}

Q4. Should v11 continue building on the v8 representation?
    {'YES -- signal exists; v11 should address calibration data and compression.' if _is_repr_sufficient else
     'CONDITIONALLY -- only if Experiment 6 confirms synthetic contamination (fixable). If the feature space itself is flat (E5 logit Cohen d < 0.2), do not continue.' if not _is_ceiling else
     'NO -- the representation does not contain sufficient information. Building another classifier on these features will not close the gap.'}

Q5. Should future work pivot to alternative approaches?
    {'Not required immediately -- continue with v8 representation but fix calibration.' if _is_repr_sufficient else
     """RECOMMENDED PIVOT PRIORITY ORDER:
    1. Attention-flow monitoring
       Directly observes how system-prompt tokens influence generation.
       Injections that hijack instruction flow alter cross-attention patterns.
       Not proxy-based -- measures the causal mechanism directly.
    2. Instruction hierarchy tracking
       Traces whether system-prompt instructions propagate correctly through
       the model's computational graph before output tokens are generated.
    3. Context consistency verification
       Compares model's final response against system-prompt constraints
       using lightweight rule or embedding comparison.
       Works post-generation; does not require architectural access.
    4. Mechanistic circuit analysis
       Identifies specific attention heads responsible for instruction
       following and monitors their activation for anomalous patterns.
       Highest signal, highest implementation cost."""}
""")
print("=" * 72)
