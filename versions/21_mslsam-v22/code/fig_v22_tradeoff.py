"""v22 figure: false alarms on reader-addressed instructions (x) vs attacks that get through (y), from metrics_v22.json."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, sys
M = json.load(open("results/v22/metrics_v22.json")); E = M["E"]; FI = ["BIPIA", "InjecAgent", "Sum-BIPIA", "Sum-TT"]
rf = lambda e: np.mean([e[f]["cross"]["reader"][0] for f in FI])
ind = lambda e: sum(e[f]["asr"][0] * e[f]["n_inj"] for f in FI) / sum(e[f]["n_inj"] for f in FI)
fig, ax = plt.subplots(figsize=(6.4, 4.4), dpi=160)
for m, col, lab in (("A0", "#2f6fd6", "A0  v20b (frozen)"), ("A1", "#d9822b", "A1  v20b + reader rows")):
    xs = [rf(E[f"{m}_s{s}"]) for s in range(5)]; ys = [ind(E[f"{m}_s{s}"]) for s in range(5)]
    ax.scatter(xs, ys, s=36, color=col, label=lab, zorder=3)
for g, mk, lab in (("promptguard2_86m", "s", "Prompt Guard 2"), ("protectai_deberta", "^", "ProtectAI v2")):
    ax.scatter(rf(E[g]), ind(E[g]), s=46, marker=mk, color="#6b7280", label=lab, zorder=3)
ax.annotate("", xy=(10.3, 60.1), xytext=(22.3, 48.3), arrowprops=dict(arrowstyle="->", color="#999", lw=1))
ax.set_xlabel("Reader-instruction false alarms, % (lower is better)")
ax.set_ylabel("Pooled indirect ASR, % (lower is better)")
ax.set_title("v22: fewer false alarms cost detection", fontsize=11, loc="left")
ax.grid(alpha=.25); ax.legend(frameon=False, fontsize=8, loc="upper right"); fig.tight_layout(); fig.savefig(sys.argv[1]); print("saved", sys.argv[1])
