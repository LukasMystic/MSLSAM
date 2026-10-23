"""MSLSAM + guard z-mean ensemble (written by v24c). ref = {mu_m, sd_m, mu_g, sd_g}: benign-logit stats on the reference rows."""
import numpy as np
def logit(p):
    p = np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def z_mean(p_mslsam, p_guard, ref):
    return ((logit(p_mslsam) - ref["mu_m"]) / ref["sd_m"] + (logit(p_guard) - ref["mu_g"]) / ref["sd_g"]) / 2
