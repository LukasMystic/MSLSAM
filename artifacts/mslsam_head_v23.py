"""MSLSAM v23 heads -- numpy only (written by v23a). kind: lr | mlr3 | mlp | mlptd."""
import numpy as np
def load(path):
    a = np.load(path, allow_pickle=False); return {k: a[k] for k in a.files}
def _pca(X, mean, comp):                       # X (n, K, d) -> (n, K*p); one BLAS product per layer
    return np.concatenate([(X[:, k] - mean[k]) @ comp[k].T for k in range(len(comp))], 1)
def features(end, anchor, art):
    end, anchor = np.asarray(end, np.float32), np.asarray(anchor, np.float32)
    F = _pca(end - anchor, art["pca_mean"], art["pca_comp"])                    # task drift  Δ = h_end - h_anchor
    if str(art["kind"]) == "mlptd": F = np.hstack([F, _pca(anchor, art["apca_mean"], art["apca_comp"])])   # + the task's own state
    return (F - art["sc_mean"]) / art["sc_scale"]
def score(end, anchor, art):
    """P(injection) per row. end, anchor: (n, K, d) hidden states at the K hooked layers."""
    F = features(end, anchor, art).astype(np.float64); k = str(art["kind"])
    if k == "lr":
        return 1.0 / (1.0 + np.exp(-(F @ art["coef"] + art["intercept"])))
    if k == "mlr3":
        z = F @ art["coef"].T + art["intercept"]; z = z - z.max(1, keepdims=True); e = np.exp(z)
        return (e / e.sum(1, keepdims=True))[:, int(art["attack_class"])]
    h = np.maximum(F @ art["W1"] + art["b1"], 0.0)                               # mlp / mlptd: one ReLU layer
    return 1.0 / (1.0 + np.exp(-(h @ art["W2"] + art["b2"]).ravel()))
