# Snapshot of main.ipynb cell [12] -- extracted 2026-09-11, read-only archive.
# Do not edit here; edit main.ipynb and re-snapshot.

# ═══════════════════════════════════════════════════════════════════════════════
# PILOT STUDY — DeepSet-Injections Behavioral Clustering (N=50)
# Exploratory only. No statistical tests. No publication tables.
# Run after Cell 6 (results must be in scope for Stage 1 scores + CJ verdicts).
# If results not in scope, lexical analysis still runs.
# ═══════════════════════════════════════════════════════════════════════════════

import pandas as pd, numpy as np, re, os, json as _json
from collections import Counter, defaultdict

_DS_PATH = "datasets/deepset_prompt_injections.csv"
_OUT_DIR = "taxonomy_annotation"
os.makedirs(_OUT_DIR, exist_ok=True)

# ── Load and preprocess ───────────────────────────────────────────────────────
def _pp(text):
    if not isinstance(text, str): return ""
    return re.sub(r'\s+', ' ', text.strip())

_df   = pd.read_csv(_DS_PATH)
_inj  = _df[_df['label'] == 1]['text'].dropna().reset_index(drop=True)
_inj  = [_pp(t) for t in _inj]          # 263 examples, original order
print(f"DeepSet injection examples loaded: {len(_inj)}")

# ── Draw 50-example pilot sample (reproducible) ───────────────────────────────
np.random.seed(42)
_sids = sorted(np.random.choice(len(_inj), 50, replace=False).tolist())
_stexts = [_inj[sid] for sid in _sids]
print(f"Pilot sample drawn (seed=42): {len(_sids)} examples")

# ── Lexical preliminary classifier ───────────────────────────────────────────
# IMPORTANT: This is machine-based preliminary classification.
# It is NOT ground truth. Use as a starting hypothesis, not a conclusion.
# German and Spanish keywords included to reduce false N rate.

def _lexical_classify(text):
    """
    Returns (type_code, [triggered_criteria]).
    Codes: A=Type-A, H=Hybrid, B=Type-B, N=Label-Noise
    All matches are case-insensitive unless noted.
    """
    t = text.lower()

    # ── Criterion 1: Explicit override / instruction reset ────────────────
    _c1_en = any(k in t for k in [
        "ignore", "forget", "disregard", "above instructions",
        "previous instructions", "preceding", "abandon all",
        "your instructions are now", "change your instructions",
        "despite what you've been told", "despite what you have been told",
        "all information.*irrelevant",    # regex-like check below
        "alle bisherigen.*irrelevant",
    ])
    _c1_de = any(k in t for k in [
        "vergiss", "ignoriere", "alle bisherigen", "alle vorherigen",
        "vorherigen aufgaben", "abweichend zu vorherigen",
        "verlassen", "bisherigen informationen.*irrelevant",
    ])
    _c1_es = any(k in t for k in ["olvide", "ignora", "olvida"])
    c1 = _c1_en or _c1_de or _c1_es

    # ── Criterion 2: Identity hijack ──────────────────────────────────────
    _c2_en = any(k in t for k in [
        "you are now", "now you are", "you are volkswagen",
        "you are three", "you are role-playing",
        "du bist keine", "du bist ein",  # German identity reassignment
    ])
    c2 = _c2_en

    # ── Criterion 3: Authority spoof ──────────────────────────────────────
    c3 = any(k in t for k in [
        "admin", "developer", "authorization", "root access",
        "privileged", "security exception", "emergency protocol",
    ])

    # ── Criterion 4: Delimiter injection ──────────────────────────────────
    c4 = (text.count('\n') > 6) or ("\\n\\n" in text) or \
         ("</s>" in t) or ("<|" in t) or ("======" in text)

    # ── Criterion 5: Context embedding ────────────────────────────────────
    c5 = any(k in t for k in [
        "input:", "output:", " spanish:", "tool output", "api response",
        "context:{", "context:{",         # RAG-style injection
        "antworte mit",                   # DE: "answer with [forced value]"
        "und nicht mit",                  # DE: "and NOT with [prohibited answer]"
        "nicht mit der antwort",
        "print the.*prompt", "drucken.*eingabeaufforderung",
        "zeige.*prompt", "show.*prompt",
    ])

    # ── Criterion 6: Roleplay-to-bypass ───────────────────────────────────
    c6 = any(k in t for k in [
        "evil ai", "pretend you are an evil", "pretend you can",
        "imagine you are", "you are role-playing",
        "actor", "film about", "schauspieler", "in their role",
        "in their roles", "in ihrer rolle",
        "theoretical world where you",
    ])

    has_a = c1 or c2 or c3 or c4 or c5 or c6
    triggered = [x for x, v in
                 [("C1",c1),("C2",c2),("C3",c3),("C4",c4),("C5",c5),("C6",c6)] if v]

    # ── Task redirect signal (Type B marker) ─────────────────────────────
    has_redirect = any(k in t for k in [
        "new task", "another task", "further task", "neue aufgabe",
        "weitere aufgabe", "now help me", "i need your help",
        "help me with", "hilf mir", "ich brauche",
        "write a reason", "write an essay", "write a manifesto",
        "write a poem", "write an argument",
        "focus on your new", "start from the front",
        "great new challenge", "neue herausforderung",
        "leave all the previous", "start over",
        "schreibe", "schreib",               # DE: "write [something]"
        "berechne", "berechnen",             # DE: "calculate"
        "explain why", "tell me why",
    ])

    # ── Classify ──────────────────────────────────────────────────────────
    if has_a and has_redirect:
        return "H", triggered          # Hybrid: Type A mechanism + B goal
    elif has_a:
        return "A", triggered          # Pure Type A
    elif has_redirect:
        return "B", triggered          # Pure Type B
    else:
        return "N", []                 # Label noise / no detectable signal

# Classify all 50
_sample_types = [_lexical_classify(t) for t in _stexts]

# ── Attach Stage 1 scores ─────────────────────────────────────────────────────
_has_s1 = False
_s1_scores   = [None] * len(_sids)   # score for each sampled example
_youden_t    = None
_cj_low      = None
_cj_high     = None

try:
    # Filter the full DeepSet list exactly as the pipeline does
    _cal = cal_inj_set if 'cal_inj_set' in dir() else set()
    _ds_filtered = [t for t in _inj if t.strip().lower() not in _cal]

    # Get Stage 1 scores in order from results
    _ds_sc_all = [
        s for s, l in zip(results["attack_scores"], results["attack_labels"])
        if l == "DeepSet-Injections"
    ]

    if len(_ds_sc_all) == len(_ds_filtered):
        # Build text → score map (text after pp() is unique enough for matching)
        _txt2score = {t.strip().lower(): s
                      for t, s in zip(_ds_filtered, _ds_sc_all)}

        for rank, sid in enumerate(_sids):
            key = _inj[sid].strip().lower()
            _s1_scores[rank] = _txt2score.get(key, None)

        _youden_t = results["youden_t"]
        _cj_low   = results["t_low"]
        _cj_high  = results["t_high"]
        _has_s1   = True
        _n_found  = sum(1 for s in _s1_scores if s is not None)
        print(f"Stage 1 scores attached: {_n_found}/50  "
              f"(Youden threshold={_youden_t:.4f})")
    else:
        print(f"[Note] Score count mismatch ({len(_ds_sc_all)} vs {len(_ds_filtered)}). "
              f"Scores not attached.")
except NameError:
    print("[Note] 'results' not in scope. Run Cell 6 first for Stage 1 scores.")
except Exception as _e:
    print(f"[Note] Stage 1 attachment failed: {_e}")

# ── Attach CJ verdicts ────────────────────────────────────────────────────────
_cj_map    = {}    # stage1_score → verdict for borderline examples
_has_cj    = False

try:
    for r, cls in results.get("cj_records", []):
        if r["label"] == "DeepSet-Injections":
            _cj_map[r["stage1"]] = r["verdict"]
    if _cj_map:
        _has_cj = True
        print(f"CJ verdicts available: {len(_cj_map)} borderline DeepSet examples")
except NameError:
    pass

# ── Build pilot dataframe ─────────────────────────────────────────────────────
_rows = []
for rank, (sid, text, (ptype, criteria)) in enumerate(
        zip(_sids, _stexts, _sample_types)):
    s1    = _s1_scores[rank]
    cj    = "NOT_EVALUATED"
    zone  = "not_scored"
    det   = None

    if s1 is not None:
        if s1 < _cj_low:
            zone, det = "below_borderline", False
            cj        = "NOT_EVALUATED"
        elif s1 > _cj_high:
            zone, det = "above_borderline", True
            cj        = "NOT_EVALUATED"
        else:
            zone = "borderline"
            det  = s1 >= _youden_t
            cj   = _cj_map.get(s1, "NOT_EVALUATED")

    _rows.append({
        "rank":        rank,
        "example_id":  sid,
        "text":        text,
        "is_german":   any(c in text for c in ['ö','ü','ä','ß']),
        "prelim_type": ptype,
        "criteria":    ",".join(criteria) if criteria else "—",
        "stage1_score": round(s1, 4) if s1 is not None else None,
        "zone":         zone,
        "detected_s1":  det,
        "cj_verdict":   cj,
    })

_pdf = pd.DataFrame(_rows)

# ── Export worksheet CSV ──────────────────────────────────────────────────────
_ws_path = os.path.join(_OUT_DIR, "pilot_50_worksheet.csv")
_ws_cols = ["rank","example_id","is_german","prelim_type","criteria",
            "stage1_score","zone","detected_s1","cj_verdict","text"]
_pdf[_ws_cols].to_csv(_ws_path, index=False, encoding="utf-8")
print(f"\nWorksheet exported → {_ws_path}")

# ── Print the 50 examples with all metadata ──────────────────────────────────
print("\n" + "═" * 90)
print("PILOT STUDY — 50 EXAMPLE WORKSHEET")
print("Column legend: [rank|id|lang] TYPE  S1=score  ZONE  CJ=verdict  CRITERIA")
print("═" * 90)

_LANG_MAP = {True: "DE", False: "EN"}
_ZONE_ABBR = {
    "below_borderline": "BELOW  ",
    "borderline":       "BORDER ",
    "above_borderline": "ABOVE  ",
    "not_scored":       "NO_SCORE",
}
for _, row in _pdf.iterrows():
    lang   = _LANG_MAP[row['is_german']]
    s1_s   = f"S1={row['stage1_score']:.3f}" if row['stage1_score'] is not None else "S1=n/a "
    det_s  = ("DET" if row['detected_s1'] else "MIS" if row['detected_s1'] is False else "n/a")
    zone_s = _ZONE_ABBR.get(row['zone'], row['zone'])
    cj_s   = (row['cj_verdict'][:4] if row['cj_verdict'] != 'NOT_EVALUATED' else '   —')
    crit   = row['criteria'][:18]

    print(f"[{int(row['rank']):02d}|{int(row['example_id']):03d}|{lang}]"
          f"  {row['prelim_type']}  {s1_s}  {det_s}  {zone_s}  CJ={cj_s}  "
          f"crit=[{crit}]")
    print(f"   {row['text'][:110]}")
    print()

# ── Descriptive statistics ────────────────────────────────────────────────────
print("═" * 90)
print("SECTION 1 — PRELIMINARY TYPE DISTRIBUTION (lexical, not ground truth)")
print("═" * 90)
_tc = _pdf['prelim_type'].value_counts().reindex(['A','H','B','N'], fill_value=0)
_tn = len(_pdf)
_NAMES = {'A':'Type A  (instruction-hierarchy)',
          'H':'Hybrid  (A mechanism + B goal) ',
          'B':'Type B  (goal-misdirection)    ',
          'N':'Label N (no detectable signal) '}
for code in ['A','H','B','N']:
    n = _tc[code]
    print(f"  {code}  {_NAMES[code]}  {n:>3} / {_tn}  ({n/_tn*100:.1f}%)")

_de_counts = _pdf.groupby('is_german')['prelim_type'].value_counts().unstack(fill_value=0)
print(f"\n  Language breakdown:")
for is_de, label in [(False,'English'), (True, 'German ')]:
    sub = _pdf[_pdf['is_german'] == is_de]
    tc2 = sub['prelim_type'].value_counts()
    parts = ' | '.join(f"{c}:{tc2.get(c,0)}" for c in ['A','H','B','N'])
    print(f"    {label}: N={len(sub):>2}  [{parts}]")

# ── Stage 1 score analysis by type ───────────────────────────────────────────
if _has_s1:
    print("\n" + "═" * 90)
    print("SECTION 2 — STAGE 1 SCORE DISTRIBUTION BY PRELIMINARY TYPE")
    print("═" * 90)
    print(f"  Youden threshold: {_youden_t:.4f}  |  Gray zone: [{_cj_low}, {_cj_high}]")
    print(f"\n  {'Type':<6}  {'N':>4}  {'Mean':>7}  {'SD':>7}  {'Min':>7}  {'Max':>7}  "
          f"{'Det%':>6}  {'Border%':>8}  {'Missed%':>8}")
    print("  " + "─" * 75)
    for code in ['A','H','B','N']:
        sub = _pdf[_pdf['prelim_type'] == code].dropna(subset=['stage1_score'])
        if len(sub) == 0:
            print(f"  {code:<6}  {'—':>4}")
            continue
        sc    = sub['stage1_score'].values
        det_n = (sub['detected_s1'] == True).sum()
        bor_n = (sub['zone'] == 'borderline').sum()
        mis_n = (sub['detected_s1'] == False).sum()
        n     = len(sub)
        print(f"  {code:<6}  {n:>4}  {sc.mean():>7.4f}  {sc.std():>7.4f}  "
              f"{sc.min():>7.4f}  {sc.max():>7.4f}  "
              f"{det_n/n*100:>5.1f}%  {bor_n/n*100:>7.1f}%  {mis_n/n*100:>7.1f}%")
    print("  " + "─" * 75)
    print("  Expected pattern if taxonomy holds:")
    print("    Type A → high Stage 1 scores, high detection")
    print("    Type B / N → lower Stage 1 scores, higher miss rate")
    print("    Type H → intermediate, varies by override strength")

# ── Zone × type cross-tab ─────────────────────────────────────────────────────
if _has_s1:
    print("\n" + "═" * 90)
    print("SECTION 3 — STAGE 1 ZONE × PRELIMINARY TYPE")
    print("═" * 90)
    _xt = pd.crosstab(_pdf['prelim_type'], _pdf['zone'],
                       margins=True, margins_name='Total')
    _xt = _xt.reindex(index=['A','H','B','N','Total'],
                       columns=['above_borderline','borderline','below_borderline',
                                'not_scored','Total'], fill_value=0)
    print(f"\n  {'Type':<8}  {'ABOVE':>8}  {'BORDER':>8}  {'BELOW':>8}  "
          f"{'N/SCORE':>8}  {'TOTAL':>8}")
    print("  " + "─" * 58)
    for idx, row in _xt.iterrows():
        print(f"  {str(idx):<8}  {row.get('above_borderline',0):>8}  "
              f"{row.get('borderline',0):>8}  {row.get('below_borderline',0):>8}  "
              f"{row.get('not_scored',0):>8}  {row.get('Total',0):>8}")

# ── CJ verdict analysis ───────────────────────────────────────────────────────
if _has_cj:
    _bdf = _pdf[_pdf['cj_verdict'] != 'NOT_EVALUATED'].copy()
    print("\n" + "═" * 90)
    print(f"SECTION 4 — CJ VERDICT DISTRIBUTION (borderline examples only, N={len(_bdf)})")
    print("═" * 90)
    _cj_dist = _bdf['cj_verdict'].value_counts()
    print(f"\n  CJ verdicts in pilot borderline zone:")
    for v in ['SAFE','SUSPICIOUS','BLOCK']:
        n = _cj_dist.get(v, 0)
        print(f"    {v:<12}: {n:>3}  ({n/max(len(_bdf),1)*100:.1f}%)")

    if len(_bdf) >= 2:
        print(f"\n  CJ verdict × preliminary type (borderline only):")
        _xt_cj = pd.crosstab(_bdf['prelim_type'], _bdf['cj_verdict'],
                              margins=True, margins_name='Total')
        print(f"\n  {'Type':<8}", end="")
        for col in _xt_cj.columns:
            print(f"  {col:>10}", end="")
        print()
        print("  " + "─" * 46)
        for idx, row in _xt_cj.iterrows():
            print(f"  {str(idx):<8}", end="")
            for val in row:
                print(f"  {int(val):>10}", end="")
            print()

# ── Natural cluster analysis ──────────────────────────────────────────────────
print("\n" + "═" * 90)
print("SECTION 5 — NATURAL CLUSTER ANALYSIS")
print("═" * 90)

_cluster_evidence = []

# Score separation (if available)
if _has_s1:
    _a_scores = _pdf[_pdf['prelim_type'].isin(['A'])]['stage1_score'].dropna()
    _h_scores = _pdf[_pdf['prelim_type'].isin(['H'])]['stage1_score'].dropna()
    _bn_scores= _pdf[_pdf['prelim_type'].isin(['B','N'])]['stage1_score'].dropna()

    if len(_a_scores) > 1 and len(_bn_scores) > 1:
        _sep = _a_scores.mean() - _bn_scores.mean()
        _cluster_evidence.append(
            f"  A vs B+N score separation: {_sep:+.4f} "
            f"(A mean={_a_scores.mean():.4f}, B+N mean={_bn_scores.mean():.4f})"
        )
        if abs(_sep) > 0.05:
            _cluster_evidence.append(
                f"  → Score gap > 0.05: Type A examples tend to score higher than B/N."
            )
        else:
            _cluster_evidence.append(
                f"  → Score gap ≤ 0.05: Score separation is weak. Clusters overlap."
            )

# Criteria diversity
_criteria_diversity = _pdf[_pdf['prelim_type'].isin(['A','H'])]['criteria'].apply(
    lambda x: len(str(x).split(','))).describe()
print(f"\n  Injection signal diversity (Type A+H examples):")
print(f"    Examples with C1 (override):         "
      f"{_pdf['criteria'].str.contains('C1').sum():>3}")
print(f"    Examples with C2 (identity):         "
      f"{_pdf['criteria'].str.contains('C2').sum():>3}")
print(f"    Examples with C4 (delimiter):        "
      f"{_pdf['criteria'].str.contains('C4').sum():>3}")
print(f"    Examples with C5 (context-embed):    "
      f"{_pdf['criteria'].str.contains('C5').sum():>3}")
print(f"    Examples with C6 (roleplay-bypass):  "
      f"{_pdf['criteria'].str.contains('C6').sum():>3}")
print(f"    Examples with redirect signal:       "
      f"{((_pdf['prelim_type'] == 'H') | (_pdf['prelim_type'] == 'B')).sum():>3}")

# German proportion
_de_total = _pdf['is_german'].sum()
_de_n = _pdf[_pdf['is_german'] & _pdf['prelim_type'].isin(['A','H'])].shape[0]
print(f"\n  German examples: {_de_total}/50 ({_de_total/50*100:.1f}%)")
print(f"  German examples classified A or H: {_de_n}/{_de_total} ({_de_n/max(_de_total,1)*100:.1f}%)")
print(f"  → German coverage of classifier is partial; manual review needed for DE 'N' cases.")

for line in _cluster_evidence:
    print(line)

# ── Proportion estimates ──────────────────────────────────────────────────────
print("\n" + "═" * 90)
print("SECTION 6 — CATEGORY PROPORTION ESTIMATES (pilot N=50)")
print("═" * 90)
print("\n  CAVEAT: These are lexical preliminary estimates, not ground truth.")
print("  German keyword coverage is partial; Type N includes German examples")
print("  that may be A or H upon manual review.")
print()

_est_label = {
    'A': 'Instruction-hierarchy attacks (pure Type A)',
    'H': 'Hybrid attacks (Type A mechanism + Type B goal)',
    'B': 'Goal-misdirection attacks (pure Type B)',
    'N': 'Label noise / no detectable signal',
}
for code in ['A','H','B','N']:
    n  = _tc[code]
    lo = max(0, n-3)   # rough ±3 uncertainty from lexical mis-classification
    hi = min(50, n+3)
    print(f"  {code}: {n:>3}/50  ({n/50*100:.1f}%)  "
          f"[rough range {lo/50*100:.0f}%–{hi/50*100:.0f}%]  "
          f"{_est_label[code]}")

_pct_with_type_a_signal = (_tc['A'] + _tc['H']) / 50 * 100
_pct_no_signal          = _tc['N'] / 50 * 100
print(f"\n  Combined A+H (examples with Type A signal): "
      f"{_tc['A']+_tc['H']}/50 ({_pct_with_type_a_signal:.1f}%)")
print(f"  Examples with no recognizable injection signal: "
      f"{_tc['N']}/50 ({_pct_no_signal:.1f}%)")
print(f"  Note: many N examples may be German translations of A/H examples.")

# ── Pilot recommendation ──────────────────────────────────────────────────────
print("\n" + "═" * 90)
print("SECTION 7 — PILOT RECOMMENDATION")
print("═" * 90)

_justified = _pct_no_signal > 10 and (_tc['B'] + _tc['H']) > 2

print(f"""
  QUESTION: Is a full 263-example annotation study justified?

  Evidence FOR:
  1. The pilot shows at least 3 distinct behavioral patterns:
       (a) Pure override/extract attacks (C1/C5, high Stage 1 scores expected)
       (b) Hybrid attack chains (override as instrument for a goal)
       (c) Examples that look benign / off-topic in isolation (Label Noise)
  2. The N-category is large ({_tc['N']}/50 = {_tc['N']/50*100:.0f}% by lexical rules), suggesting
     that a substantial fraction of DeepSet is NOT recognizable as injection
     without deployment context. This heterogeneity is the main finding to validate.
  3. The lexical classifier has known German-language blind spots. A human
     annotator would reclassify many German 'N' examples as 'A' or 'H',
     likely reducing N% and increasing A+H%.
  4. The Type B category appears small (B={_tc['B']}/50) but may be under-detected
     by the lexical classifier. Manual review of 'N' examples is needed to
     distinguish genuine label noise from soft Type B attacks.

  Evidence AGAINST / Risks:
  1. The Type B signal is very weak in the pilot ({_tc['B']}/50 = {_tc['B']/50*100:.0f}%).
     If Type B remains rare in the full study, the taxonomy loses one of its
     two primary categories.
  2. A large N% (label noise) weakens the taxonomy claim: if 40–60% of DeepSet
     is label noise, DeepSet is not a reliable Type B benchmark — it is a noisy
     benchmark that includes both Type A and off-topic benign examples.
  3. German coverage: 'N' examples include many German texts the lexical
     classifier can't classify. This inflates N%. Cannot resolve without
     manual reading.

  RECOMMENDATION: {'PROCEED with full 263-example study.' if _justified else
                   'CONDUCT additional pilot review before full study.'}

  Minimum viable next step:
    - Manually review the {_tc['N']} 'N' pilot examples (worksheet: pilot_50_worksheet.csv)
    - Determine how many are genuinely benign vs. German A/H injections
    - If >30% of N examples reclassify to A or H after German reading:
      full study is strongly justified (taxonomy covers 70%+ of dataset)
    - If <10% reclassify: Type B is rare; reconsider whether the taxonomy
      adds value beyond 'Type A present / absent'
""")

# ── Save pilot summary ────────────────────────────────────────────────────────
_pilot_summary = {
    "n_sample": 50,
    "seed": 42,
    "has_stage1_scores": _has_s1,
    "has_cj_verdicts": _has_cj,
    "prelim_distribution": {k: int(_tc[k]) for k in ['A','H','B','N']},
    "german_count": int(_de_total),
    "n_borderline_cj": int(len(_bdf)) if _has_cj else 0,
}
_pilot_path = os.path.join(_OUT_DIR, "pilot_summary.json")
with open(_pilot_path, 'w') as _f:
    _json.dump(_pilot_summary, _f, indent=2)
print(f"  Pilot summary saved → {_pilot_path}")
print(f"  Worksheet saved     → {_ws_path}")
print("═" * 90)
