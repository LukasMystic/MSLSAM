"""
v14d - post-hoc analysis of v14c run 2 (HistGradientBoost).

Not part of the notebook pipeline. Reads results/v14/oof_full.csv and the corpus,
and answers two questions the aggregate metrics cannot:

  1. Does the detector find the PAYLOAD, or just recognise the CARRIER DOCUMENT?
     -> paired margin between each injected row and its own clean carrier.
  2. Is the per-corpus FPR blowout a detection failure or a THRESHOLD failure?
     -> per-context 10%-FPR thresholds, and the ASR they imply.

Run:  python v14d_posthoc_threshold.py
Written 2026-09-27. Read-only; writes nothing.
"""
import pandas as pd, numpy as np, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]          # Experiments/
CORPUS = ROOT / "datasets/unified/injection_corpus.csv"
OOF    = ROOT / "results/v14/oof_full.csv"

df  = pd.read_csv(CORPUS)
oof = pd.read_csv(OOF)
ev  = df[df.family != "baseline"].reset_index(drop=True)
assert len(ev) == len(oof), "row count mismatch"
assert (ev.payload_id.astype(str).values == oof.payload_id.astype(str).values).all(), \
    "oof_full.csv row order does not match the evaluation frame"
ev = ev.copy(); ev["score"] = oof.score.values

# ---------------------------------------------------------------- 1. payload vs carrier
# NOTE: carrier_id CANNOT be used to join here. v14a hashes it with the corpus name,
# so an injected row gets "BIPIA:<h>" and its clean twin "BIPIA-Clean:<h2>" - different
# keys for the same document. FIX THIS IN v14a. Matching by text prefix as a workaround.
print("=" * 72)
print("1. PAIRED CARRIER MARGIN  (does it find the payload?)")
print("=" * 72)
for c in ["BIPIA", "InjecAgent"]:
    inj, cln = ev[ev.corpus == c], ev[ev.corpus == c + "-Clean"]
    cl, cs = cln.text.astype(str).tolist(), cln.score.tolist()
    margins = []
    for _, r in inj.iterrows():
        t = str(r.text)
        for ct, csc in zip(cl, cs):
            if ct[:120] and ct[:120] in t:
                margins.append(r.score - csc); break
    pos = sum(1 for m in margins if m > 0) / len(margins) * 100
    print(f"\n{c}: {len(margins)}/{len(inj)} injected rows matched to their clean carrier")
    print(f"  mean score  injected {inj.score.mean():.4f} | clean carrier {cln.score.mean():.4f}")
    print(f"  paired margin  mean {st.mean(margins):+.4f}  median {st.median(margins):+.4f}")
    print(f"  injected scored above its own clean carrier: {pos:.1f}%   (50% = no payload signal)")

# ---------------------------------------------------------------- 2. threshold placement
print("\n" + "=" * 72)
print("2. THRESHOLD PLACEMENT  (is the FPR blowout calibration, not detection?)")
print("=" * 72)
ben = ev[ev.label == 0]
thr = np.quantile(ben.score, 0.90)
print(f"\nglobal 90th-pct benign threshold = {thr:.6f}\n")
g = ben.groupby("corpus").score.agg(["count", "mean", "median",
                                     lambda s: (s > thr).mean() * 100])
g.columns = ["n", "mean", "median", "FPR%"]
print(g.sort_values("FPR%", ascending=False).round(4))

print("\nthreshold each benign context would need for its OWN 10% FPR:")
for c, s in ben.groupby("corpus").score:
    print(f"  {c:22s} n={len(s):5d}  local-90th {np.quantile(s, 0.90):.6f}")

# ---------------------------------------------------------------- 3. context-matched ASR
print("\n" + "=" * 72)
print("3. CONTEXT-MATCHED OPERATING POINT")
print("   every benign context held to 10% FPR; ASR measured against its own context")
print("   OPTIMISTIC: threshold fitted on the same benign rows it is scored on.")
print("   v15 must nest this (calibration half / measurement half) before citing.")
print("=" * 72 + "\n")
pairs = {"BIPIA": "BIPIA-Clean", "InjecAgent": "InjecAgent-Clean",
         "SPML": "SPML-Benign",  "DeepSet": "DeepSet-Benign"}
ti = tm = 0
for inj, benc in pairs.items():
    si = ev[ev.corpus == inj].score.values
    sb = ev[ev.corpus == benc].score.values
    t = np.quantile(sb, 0.90); miss = int((si <= t).sum())
    ti += len(si); tm += miss
    print(f"  {inj:11s} thr {t:.6f}  ASR {miss/len(si)*100:6.2f}%  ({miss}/{len(si)})"
          f"   FPR {100*(sb>t).mean():5.1f}% on {benc} (n={len(sb)})")
print(f"\n  POOLED context-matched ASR: {tm/ti*100:.2f}%  ({tm}/{ti})")
print(f"  (global single threshold reported ASR 3.25% at a pooled FPR that hides")
print(f"   InjecAgent-Clean 100.0% and BIPIA-Clean 47.0%)")
