"""v24 figure redrawn from results/v24/metrics_v24.json (legend below the bars)."""
import json, sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
E = json.load(open("results/v24/metrics_v24.json"))["E"]; F = ["News-GS", "Table-MC"]
pooled = lambda e: sum(e[f]["asr"][0] * e[f]["n_inj"] for f in F) / sum(e[f]["n_inj"] for f in F)
def v(cols): return [np.mean([E[c][f]["asr"][0] for c in cols]) for f in F] + [np.mean([pooled(E[c]) for c in cols])]
S = lambda g: [f"{g}_s{s}" for s in range(5)]
M = [("MSLSAM v20b", v(S("A0")), "#2f6fd6"), ("MSLSAM + tables (descr.)", v(S("A2")), "#8fb3e8"), ("Prompt Guard 2", v(["promptguard2_86m"]), "#9aa3af"),
     ("ProtectAI v2", v(["protectai_deberta"]), "#c4c9d0"), ("Ensemble v20b + PG2 (frozen)", v(S("ENS0")), "#d9822b"), ("Ensemble + tables (descr.)", v(S("ENS2")), "#f2b27a")]
x = np.arange(3); w = 0.13; fig, ax = plt.subplots(figsize=(7.6, 4.2), dpi=160)
for i, (n, val, c) in enumerate(M):
    b = ax.bar(x + (i - 2.5) * w, val, w, label=n, color=c)
    for r, y in zip(b, val): ax.text(r.get_x() + r.get_width() / 2, y + 0.4, f"{y:.1f}", ha="center", fontsize=6)
ax.axhline(5, color="#b91c1c", lw=0.8, ls="--"); ax.text(2.45, 5.5, "target 5%", color="#b91c1c", fontsize=7, ha="right")
ax.set_xticks(x, F + ["pooled"]); ax.set_ylabel("ASR % at 10% FPR (lower is better)"); ax.grid(axis="y", alpha=.25)
ax.set_title("v24 fresh confirmation (mean of 5 seeds)", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.09)); fig.tight_layout(); fig.savefig(sys.argv[1]); print("saved")
