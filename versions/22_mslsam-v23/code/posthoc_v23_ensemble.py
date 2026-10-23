"""v23 POST HOC (not pre-registered, descriptive): v21's pre-specified z-mean ensemble (v20b + Prompt Guard 2, mu/sigma frozen in v21) on the v23 fresh set; Table-TTX misses by position; the Clean+Reader calibration view. Run from Experiments/."""
import json, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import binomtest
M = json.load(open("results/v23/metrics_v23.json"))
print("TARGET worst-seed:", {k: {kk: round(v, 2) if isinstance(v, float) else v for kk, v in d.items()} for k, d in M["target"].items()})
S = pd.read_csv("results/v23/all_scores_v23.csv")
nb = json.load(open("v23e_confirm.ipynb")); src = "".join(nb["cells"][2]["source"])
g = {"S": S, "np": np, "pd": pd, "roc_auc_score": roc_auc_score, "average_precision_score": average_precision_score, "binomtest": binomtest,
     "N_THRESH": 400, "FPR_BUDGET": 0.10}
exec(src, g)
# v21's PRE-SPECIFIED z-mean ensemble, with v21's frozen references (no v23 data used to set mu/sigma)
ext = pd.read_csv("datasets/unified/v19_external_train.csv", keep_default_na=False); tool = pd.read_csv("datasets/unified/v20_tool_train.csv", keep_default_na=False)
yx = np.r_[ext.label.values, tool.label.values]
REF = {int(k): v for k, v in json.load(open("artifacts/mslsam_v20b_ref.json")).items()}
PG_REF = np.load("results/v21/scores/promptguard2_86m__ref.npy").astype(np.float64)
logit = lambda p: (lambda q: np.log(q / (1 - q)))(np.clip(np.asarray(p, np.float64), 1e-6, 1 - 1e-6))
lg = logit(PG_REF[yx == 0]); ZG = (lg.mean(), lg.std() + 1e-9)
for s in range(5):
    S[f"ENS_s{s}"] = ((logit(S[f"A0_s{s}"]) - REF[s]["mu"]) / REF[s]["sd"] + (logit(S["promptguard2_86m"]) - ZG[0]) / ZG[1]) / 2
g["S"] = S; g["cor"] = S.corpus.values
E = {c: g["evaluate"](c) for c in [f"A0_s{s}" for s in range(5)] + [f"ENS_s{s}" for s in range(5)] + ["promptguard2_86m", "protectai_deberta"]}
g["E"] = E
for m in ("A0", "ENS"):
    rows = []
    for s in range(5):
        e = E[f"{m}_s{s}"]; w, l = [], []
        for f in g["FAMS"]:
            r = g["mcnemar"](f"{m}_s{s}", "promptguard2_86m", f)
            (w if r["winner"] == "a" else l if r["winner"] == "b" else []).append(f)
        rows.append({"seed": s, "Review": e["Review-TTX"]["asr"][0], "Table": e["Table-TTX"]["asr"][0], "IND": g["IND"](e), "RF": g["RF"](e),
                     "OffTask_R": e["Review-TTX"]["cross"]["offtask"][0], "OffTask_T": e["Table-TTX"]["cross"]["offtask"][0], "mc_wins_vs_PG2": w, "mc_losses_vs_PG2": l,
                     "worstFPR": max(e[f]["fpr"][0] for f in g["FAMS"])})
    print(f"\n== {m} (post hoc) ==\n" + pd.DataFrame(rows).round(2).to_string(index=False))
for c in ("promptguard2_86m", "protectai_deberta"):
    e = E[c]; print(c, {f: round(e[f]["asr"][0], 2) for f in g["FAMS"]}, "IND", round(g["IND"](e), 2), "RF", round(g["RF"](e), 2))
# by position and by payload: where does A0 miss on tables?
F = pd.read_csv("datasets/unified/v23_fresh.csv", keep_default_na=False)
t0 = E["A0_s0"]["Table-TTX"]["threshold"][0]; tp = E["promptguard2_86m"]["Table-TTX"]["threshold"][0]
m = (F.corpus == "Table-TTX").values
miss_a, miss_p = S["A0_s0"].values[m] < t0, S["promptguard2_86m"].values[m] < tp
print("\nTable-TTX by position (A0 s0 miss % | PG2 miss %):", {p: (round(100 * miss_a[F.pos.values[m] == p].mean(), 1), round(100 * miss_p[F.pos.values[m] == p].mean(), 1)) for p in ("start", "middle", "end")})
print("A0 misses, PG2 catches -- examples:")
for t in F.piece.values[m][miss_a & ~miss_p][:6]: print("   ", t[:140])
print("Table carriers -- A0 score on Clean tier median %.3f vs Review Clean %.3f" % (S["A0_s0"][S.corpus == "Table-TTX-Clean"].median(), S["A0_s0"][S.corpus == "Review-TTX-Clean"].median()))
print("\n== secondary: threshold on Clean + Reader together ==")
def pb(col, n_rep=200):
    s_ = S[col].values; rng_ = np.random.default_rng(0); ti = tm = 0; out = {}
    for f in g["FAMS"]:
        cal = np.r_[s_[g["cor"] == f + "-Clean"], s_[g["cor"] == f + "-Reader"]]; inj = s_[g["cor"] == f]
        a = [(inj < g["thr"](cal[rng_.permutation(len(cal))[:len(cal) // 2]])).mean() * 100 for _ in range(n_rep)]
        out[f] = np.mean(a); ti += len(inj); tm += np.mean(a) / 100 * len(inj)
    return out, tm / ti * 100
for c in ["A0_s0", "ENS_s0", "ENS_s1", "ENS_s2", "ENS_s3", "ENS_s4", "promptguard2_86m", "protectai_deberta"]:
    o, p = pb(c); print(f"  {c:<18} " + " | ".join(f"{f} {v:.2f}" for f, v in o.items()) + f" | pooled {p:.2f}")
ens = [g["IND"](E[f"ENS_s{s}"]) for s in range(5)]; n = sum(E["ENS_s0"][f]["n_inj"] for f in g["FAMS"])
print("ensemble pooled ASR mean %.2f, worst %.2f, Wilson CI (worst seed) %s, n_inj %d" % (np.mean(ens), max(ens), g["wilson"](int(round(max(ens) / 100 * n)), n), n))
