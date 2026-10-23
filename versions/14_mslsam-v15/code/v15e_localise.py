# v15e - the decisive question behind v15's negative result:
# IS AN INJECTED PAYLOAD A LOCALISED DISTURBANCE IN THE TRAJECTORY AT ALL?
#
# v15c's Exp 3 sat on the majority baseline. v15d showed why the "all" number is
# uninformative: BIPIA places half its payloads at segment 0, and D_0 = Z_0 - MU is a
# magnitude vector, not a step. Across rows, D_0 vectors align at cos ~0.24-0.36 while
# step vectors align at ~0.04 -- so the max-cosine argmax is dragged to segment 0, and
# "exact" tracks the share of start-payloads rather than any localisation.
#
# This script uses ONLY end-payload rows (no D_0 in ground truth or prototypes) and
# compares three localisers against a CONSTANT predictor:
#   1. argmax of max-cosine to grouped-out payload prototypes   (v15c's statistic)
#   2. argmax of ||D_k||   -- "the payload is the biggest state change", no training
#   3. constant = modal segment                                  -- the baseline to beat
# If none beats the constant predictor, a payload is not a localised trajectory
# disturbance in this representation, and v15's premise fails for the right reason.
import numpy as np, pandas as pd, os, json

CACHE="representation_cache/chunked_v15"
rows=pd.read_csv(f"{CACHE}/rows.csv"); man=json.load(open(f"{CACHE}/manifest.json"))
MC=man["max_chunks"]
SH=sorted(f for f in os.listdir(CACHE) if f.startswith("shard_"))
IDX=[np.load(f"{CACHE}/{f}")["row_idx"] for f in SH]
def load_layer(li):
    X=np.zeros((len(rows),MC,4096),np.float16)
    for f,idx in zip(SH,IDX): X[idx]=np.load(f"{CACHE}/{f}")[f"layer_{li}"]
    return X
rows["pos"]=rows.meta.astype(str).str.extract(r"pos=(\w+)")
NCH=rows.n_chunks.values.astype(int)
def _norm(A): return A/(np.linalg.norm(A,axis=-1,keepdims=True)+1e-8)

b=rows[(rows.corpus=="BIPIA")&(rows.pos=="end")].copy()
idx=b.index.values; k=NCH[idx]
frac=b.span_start.values/np.maximum(1,b.n_chars.values)
gt=np.minimum(k-1,np.maximum(0,(frac*k).astype(int)))
vm=np.arange(MC)[None,:]<k[:,None]
vals,cnts=np.unique(gt,return_counts=True); MODE=int(vals[cnts.argmax()])
const=np.full(len(gt),MODE)
print(f"BIPIA end-payload rows: {len(b)} | payload segment distribution: {dict(zip(vals.tolist(),cnts.tolist()))}")
print(f"constant predictor = segment {MODE}\n")

def score(pred,name):
    return (name,(pred==gt).mean()*100,(np.abs(pred-gt)<=1).mean()*100)

grp=b.payload_id.values; ug=np.array(sorted(set(grp)))
rng=np.random.default_rng(42); rng.shuffle(ug)
fold={g:i%5 for i,g in enumerate(ug)}; fa=np.array([fold[g] for g in grp])

print(f"{'layer':>6}  {'localiser':<26}{'exact %':>9}{'within1 %':>11}{'vs const':>10}")
print("-"*64)
base=score(const,"constant (modal segment)")
for li in [4,8,16,24,31]:
    X=load_layer(li).astype(np.float32)
    isb=(rows.corpus=="Baseline-Dolly").values
    MUb=X[isb].reshape(-1,4096)[(np.arange(MC)[None,:]<NCH[isb][:,None]).reshape(-1)].mean(0)
    D=np.concatenate([X[:,0:1,:]-MUb[None,None,:],np.diff(X,axis=1)],axis=1)
    # 1. max-cosine to grouped-out prototypes
    pred_c=np.zeros(len(idx),int)
    for f in range(5):
        te=np.where(fa==f)[0]; tr=np.where(fa!=f)[0]
        if not len(te) or not len(tr): continue
        P=D[idx[tr],gt[tr]]
        S=_norm(D[idx[te]])@_norm(P).T
        pred_c[te]=np.where(vm[te],S.max(axis=2),-np.inf).argmax(1)
    # 2. argmax ||D_k||, no training at all
    pred_n=np.where(vm,np.linalg.norm(D[idx],axis=2),-np.inf).argmax(1)
    for nm,pr in (("max-cos to prototypes",pred_c),("argmax ||D_k|| (untrained)",pred_n)):
        _,e,w=score(pr,nm)
        print(f"{li:>6}  {nm:<26}{e:>8.1f}%{w:>10.1f}%{e-base[1]:>+9.1f}")
    del X,D
print("-"*64)
print(f"{'':>6}  {base[0]:<26}{base[1]:>8.1f}%{base[2]:>10.1f}%{0.0:>+9.1f}")
