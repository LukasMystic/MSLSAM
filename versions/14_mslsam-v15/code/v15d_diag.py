# v15d - diagnostic on v15c's negative result. numpy/pandas only (no sklearn on this VM),
# so it works in RAW 4096-d space instead of PCA space. Cosine similarity is unaffected by
# a linear projection's presence in the way that matters here, so this is a valid proxy.
#
# QUESTION: v15c's Exp 3 localisation (48.7%) sits almost exactly on the majority baseline
# (43.0%), and the 43% is the share of BIPIA payloads placed at SEGMENT 0. Suspicion:
# the segment-0 fix I made mixes two DIFFERENT KINDS of vector in one prototype pool.
#   D_0 = Z_0 - MU        (a magnitude: state relative to the baseline mean)
#   D_k = Z_k - Z_(k-1)   (a step: what this segment added)
# Those point in systematically different directions. With ~43% of prototypes being
# D_0-type, every query's own segment-0 delta matches them strongly and the argmax is
# dragged to segment 0 -- which would reproduce the majority baseline and look like
# "no localisation" whether or not the mechanism works.
#
# TEST: restrict to BIPIA payloads placed at the END. Those never sit at segment 0, so
# scoring them against a pool of END-payload prototypes involves step vectors only. If
# localisation there is well above its own majority baseline, the mechanism works and the
# D_0 mixing is the confound. If it is still at baseline, position features genuinely do
# not carry the signal.
import numpy as np, pandas as pd, os, json

CACHE = "representation_cache/chunked_v15"
rows  = pd.read_csv(f"{CACHE}/rows.csv")
man   = json.load(open(f"{CACHE}/manifest.json"))
MC, LAYERS = man["max_chunks"], man["layers"]
SH  = sorted(f for f in os.listdir(CACHE) if f.startswith("shard_"))
IDX = [np.load(f"{CACHE}/{f}")["row_idx"] for f in SH]

def load_layer(li):
    X = np.zeros((len(rows), MC, 4096), np.float16)
    for f, idx in zip(SH, IDX):
        X[idx] = np.load(f"{CACHE}/{f}")[f"layer_{li}"]
    return X

rows["pos"] = rows.meta.astype(str).str.extract(r"pos=(\w+)")
NCH = rows.n_chunks.values.astype(int)

def payload_seg(sub):
    k = NCH[sub.index.values]
    frac = (sub.span_start.values / np.maximum(1, sub.n_chars.values))
    return np.minimum(k - 1, np.maximum(0, (frac * k).astype(int)))

def _norm(A): return A / (np.linalg.norm(A, axis=-1, keepdims=True) + 1e-8)

def localise(D, sub, gt, folds=5, seed=42):
    # grouped-out prototypes by payload_id, self column masked -- same protocol as Exp 3
    idx = sub.index.values
    grp = sub.payload_id.values
    ug  = np.array(sorted(set(grp)))
    rng = np.random.default_rng(seed); rng.shuffle(ug)
    fold = {g: i % folds for i, g in enumerate(ug)}
    fa = np.array([fold[g] for g in grp])
    pred = np.zeros(len(idx), int)
    for f in range(folds):
        te = np.where(fa == f)[0]; tr = np.where(fa != f)[0]
        if not len(te) or not len(tr): continue
        P = D[idx[tr], gt[tr]]                       # prototype = delta AT the payload segment
        S = _norm(D[idx[te]].astype(np.float32)) @ _norm(P.astype(np.float32)).T
        vm = np.arange(MC)[None, :] < NCH[idx[te]][:, None]
        pred[te] = np.where(vm, S.max(axis=2), -np.inf).argmax(1)
    return pred

print("=" * 76)
print("v15d - is the segment-0 delta mixing two kinds of vector?")
print("=" * 76)

b = rows[rows.corpus == "BIPIA"].copy()
print(f"\nBIPIA injected: {len(b)}  | pos: {b.pos.value_counts().to_dict()}")

for li in [4, 16, 31]:
    X = load_layer(li).astype(np.float32)
    MU = X[(rows.corpus == "Baseline-Dolly").values].reshape(-1, 4096)
    keep = (np.arange(MC)[None, :] < NCH[(rows.corpus == "Baseline-Dolly").values][:, None]).reshape(-1)
    MU = MU[keep].mean(0)
    D = np.concatenate([X[:, 0:1, :] - MU[None, None, :], np.diff(X, axis=1)], axis=1)
    print(f"\n--- layer {li} ---")
    # direction check: are D_0 and D_k systematically different?
    d0 = _norm(D[:, 0, :]); dk = _norm(D[:, 3, :])
    print(f"  mean cos(D_0 of row i, D_0 of row j) = {float((d0[:200] @ d0[:200].T)[np.triu_indices(200,1)].mean()):+.3f}")
    print(f"  mean cos(D_3 of row i, D_3 of row j) = {float((dk[:200] @ dk[:200].T)[np.triu_indices(200,1)].mean()):+.3f}")
    print(f"  mean cos(D_0, D_3) same row          = {float((d0[:2000]*dk[:2000]).sum(1).mean()):+.3f}")
    # ALSO: score end-payloads against END-ONLY prototypes (step vectors only, no D_0)
    for tag in ["all", "end", "start"]:
        sub = b if tag == "all" else b[b.pos == tag]
        if not len(sub): continue
        gt = payload_seg(sub)
        pred = localise(D, sub, gt)
        vals, cnts = np.unique(gt, return_counts=True)
        maj = cnts.max() / cnts.sum() * 100
        ex  = (pred == gt).mean() * 100
        w1  = (np.abs(pred - gt) <= 1).mean() * 100
        print(f"  {tag:<6} n={len(sub):5d}  exact {ex:5.1f}%  within1 {w1:5.1f}%  "
              f"majority {maj:5.1f}%  lift {ex-maj:+6.1f}")
    del X, D
