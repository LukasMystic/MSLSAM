"""v24 POST HOC (descriptive): Table-MC misses by attack length; examples ENS0 misses. Run from Experiments/."""
import json, numpy as np, pandas as pd
S = pd.read_csv("results/v24/all_scores_v24.csv"); F = pd.read_csv("datasets/unified/v24_fresh.csv", keep_default_na=False)
E = json.load(open("results/v24/metrics_v24.json"))["E"]
m = (F.corpus == "Table-MC").values; L = F.piece.str.len().values
q = np.quantile(L[m], [1/3, 2/3]); bins = np.digitize(L, q)
print(f"Table-MC attack length terciles: <{q[0]:.0f}, {q[0]:.0f}-{q[1]:.0f}, >{q[1]:.0f} chars")
for c in ("A0_s0", "A2_s0", "ENS0_s0", "ENS2_s0", "promptguard2_86m", "protectai_deberta"):
    t = E[c]["Table-MC"]["threshold"][0]; miss = S[c].values < t
    print(f"  {c:<18} miss %: " + " | ".join(f"{100 * miss[m & (bins == b)].mean():5.1f}" for b in range(3)))
t = E["ENS0_s0"]["Table-MC"]["threshold"][0]; miss = (S["ENS0_s0"].values < t) & m
print("ENS0 s0 misses on Table-MC (examples):")
for p in F.piece.values[miss][:10]: print("   ", p[:110])
