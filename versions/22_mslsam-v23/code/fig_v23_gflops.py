"""v23 figure: GFLOPs per prompt (median of the v23 fresh prompts), redrawn from results/v23/flops_v23.json with a title that fits."""
import json, sys, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
T = json.load(open("results/v23/flops_v23.json"))["table"]
T = [r for r in T if "candidate" not in r["component"] or "MLPTD" in r["component"]]
ren = {"Llama-3-8B prefill (the app's own pass; shared)": "Llama-3-8B prefill\n(the app's own pass)", "8B judge (a second prefill + 1 answer position)": "8B judge (2nd prefill)",
       "Prompt Guard 2 (window-max)": "Prompt Guard 2", "ProtectAI v2 (window-max)": "ProtectAI v2", "MSLSAM v20b (A0)": "MSLSAM head (v20b)",
       "v23 candidate MLPTD": "MSLSAM head (MLP-TD)"}
lab = [ren.get(r["component"], r["component"]) for r in T]; val = [r["GFLOPs median"] for r in T]
fig, ax = plt.subplots(figsize=(7.2, 3.4), dpi=160)
ax.barh(lab[::-1], val[::-1], color=["#2f6fd6" if "MSLSAM" in l else "#9aa3af" for l in lab[::-1]])
ax.set_xscale("log"); ax.set_xlim(5e-3, 4e4)
for y, v in enumerate(val[::-1]): ax.text(v * 1.2, y, f"{v:,.0f}" if v >= 10 else f"{v:.3f}", va="center", fontsize=7)
ax.set_xlabel("GFLOPs per prompt (median, log scale)"); ax.set_title("Compute per prompt, v23 fresh prompts (median 372 tokens)", fontsize=10, loc="left")
fig.tight_layout(); fig.savefig(sys.argv[1]); print("saved")
