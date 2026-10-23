"""v25 POST HOC (descriptive, not pre-registered): does the layer-12 single-layer probe keep its advantage when Reader rows count as
benign (threshold on Clean + Reader)? And AUC saturation: ASR spread among cells with AUC >= 0.95. Run from Experiments/."""
import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import binomtest
S = pd.read_csv("results/v25/all_scores_v25.csv")
nb = json.load(open("v25_layers_calibration.ipynb")); g = {"S": S, "np": np, "pd": pd, "roc_auc_score": roc_auc_score,
     "average_precision_score": average_precision_score, "binomtest": binomtest, "N_THRESH": 400, "FPR_BUDGET": 0.10}
exec("".join(nb["cells"][2]["source"]), g); cor = S.corpus.values; thr = g["thr"]; FRESH = g["FRESH_F"]
def pb(col, n_rep=200):
    s_ = S[col].values; rng = np.random.default_rng(0); ti = tm = 0; per = {}
    for f in FRESH:
        cal = np.r_[s_[cor == f + "-Clean"], s_[cor == f + "-Reader"]]; inj = s_[cor == f]; a = []; fr = []
        for _ in range(n_rep):
            i = rng.permutation(len(cal)); t = thr(cal[i[:len(cal) // 2]]); a.append((inj < t).mean() * 100)
        per[f] = np.mean(a); ti += len(inj); tm += np.mean(a) / 100 * len(inj)
    return tm / ti * 100, per
rows = {}
for nm, cols in (("A0 (8 layers)", [f"A0_s{s}" for s in range(5)]), ("ENS0 (A0 + PG2)", [f"ENS0_s{s}" for s in range(5)]),
                 ("B1 L12 (TaskTracker type)", ["B1_L12"]), ("B2 L12 (our recipe)", [f"B2_L12_s{s}" for s in range(5)]),
                 ("C EARLY {4,8,12}", [f"EARLY_s{s}" for s in range(5)]), ("Prompt Guard 2", ["promptguard2_86m"]), ("ProtectAI", ["protectai_deberta"])):
    r = [pb(c) for c in cols]; rows[nm] = {"pooled (Clean+Reader cal.)": np.mean([x[0] for x in r]), **{f: np.mean([x[1][f] for x in r]) for f in FRESH}}
print("POST HOC: fresh ASR % with the threshold set on Clean + Reader (reader instructions count as benign)")
print(pd.DataFrame(rows).T.round(2).to_string())
M = json.load(open("results/v25/metrics_v25.json")); C = pd.DataFrame(M["D"]["cells"])
for lo, hi in ((0.95, 0.97), (0.97, 0.99), (0.99, 1.01)):
    d = C[(C.auc >= lo) & (C.auc < hi)]; print(f"AUC [{lo:.2f}, {hi:.2f}): {len(d)} cells, ASR range {d.asr.min():.1f}-{d.asr.max():.1f} (median {d.asr.median():.1f})")
