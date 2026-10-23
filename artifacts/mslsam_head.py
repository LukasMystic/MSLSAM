"""MSLSAM-Δ head -- numpy only. This is ALL a deployment needs besides the LLM's hooks (written by v21a)."""
import numpy as np
def load(path):
    a = np.load(path, allow_pickle=False); return {k: a[k] for k in a.files}
def features(end, anchor, art):
    """end, anchor: (n, K, d) hidden states at the K hooked layers (art['layers'] order) -> (n, K*k) scaled features."""
    D = np.asarray(end, np.float32) - np.asarray(anchor, np.float32)              # task drift  Δ = h_end - h_anchor
    F = np.einsum("nkd,kpd->nkp", D - art["pca_mean"][None], art["pca_comp"]).reshape(len(D), -1)
    return (F - art["sc_mean"]) / art["sc_scale"]
def score(end, anchor, art):
    """P(injection) for each row. Cost: K*k*d multiply-adds (8*256*4096 = 8.4 M), independent of prompt length."""
    z = features(end, anchor, art) @ art["coef"] + art["intercept"]
    return 1.0 / (1.0 + np.exp(-z.astype(np.float64)))
