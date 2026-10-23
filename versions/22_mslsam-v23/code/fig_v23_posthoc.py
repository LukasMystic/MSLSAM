"""v23 figure (POST HOC): ASR on the v23 fresh families at each family's 10%-FPR clean threshold -- v20b, the guards, and v21's
pre-specified z-mean ensemble (v20b + Prompt Guard 2, mu/sigma frozen in v21). Values from results/posthoc_v23_ensemble.txt (seed means)."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np, sys
vals = {"MSLSAM v20b": (2.62, 14.82, 9.47), "Prompt Guard 2": (6.56, 4.77, 5.55), "ProtectAI v2": (16.32, 7.33, 11.27),
        "v20b + PG2 ensemble\n(post hoc)": (1.05, 0.74, 0.88)}
cols = ["#2f6fd6", "#9aa3af", "#c4c9d0", "#d9822b"]; fams = ["Review-TTX", "Table-TTX", "pooled"]; x = np.arange(3); w = 0.2
fig, ax = plt.subplots(figsize=(7.2, 3.8), dpi=160)
for i, (k, v) in enumerate(vals.items()):
    b = ax.bar(x + (i - 1.5) * w, v, w, label=k, color=cols[i])
    for r, val in zip(b, v): ax.text(r.get_x() + r.get_width() / 2, val + 0.3, f"{val:.1f}", ha="center", fontsize=7)
ax.axhline(5, color="#b91c1c", lw=0.8, ls="--"); ax.text(2.45, 5.3, "target 5%", color="#b91c1c", fontsize=7, ha="right")
ax.set_xticks(x, fams); ax.set_ylabel("ASR % at 10% FPR (lower is better)")
ax.set_title("v23 fresh data: the two detectors miss different attacks", fontsize=10, loc="left")
ax.legend(frameon=False, fontsize=7, ncol=2); ax.grid(axis="y", alpha=.25); fig.tight_layout(); fig.savefig(sys.argv[1]); print("saved")
