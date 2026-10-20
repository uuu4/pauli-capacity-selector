"""Kati altin, Deney 1 (Faz B v3): A6_wnorm etiketsiz kollardan cikarilinca (makale 2026-10-20: wnorm anilmiyor,
merge tek cumleyle tanimli). syn52.py ile ayni tanim ve bootstrap; yalniz UNS'den A6_wnorm cikti.
Calistir (runs/ icinden): python selection_bias_check_v1/syn52c.py"""
import json, numpy as np, collections
rng = np.random.default_rng(0)
def boot(rows, f, B=4000):
    cells = sorted({r['_u'] for r in rows}); by = collections.defaultdict(list)
    for r in rows: by[r['_u']].append(f(r))
    out = []
    for _ in range(B):
        s = rng.choice(len(cells), len(cells)); out.append(np.median(np.concatenate([by[cells[i]] for i in s])))
    return np.percentile(out, [2.5, 97.5])
d = json.load(open('phase_bv3_full_v1/phase_bv3_FULL_results.json')); rows = d['rows']
for r in rows: r['_u'] = (r['n'], r['depth'], r['ent'], 0.3, 1, 0.0)
for name, UNS in [('wnorm dahil (syn52)', ('A1_random', 'A2_hbody', 'A3_topp', 'A4_merge', 'A6_wnorm', 'A7_resid')),
                  ('wnorm haric', ('A1_random', 'A2_hbody', 'A3_topp', 'A4_merge', 'A7_resid'))]:
    g = [r for r in rows if r['label'] != 'qlocal' and r['A0'] < 53 and max(r[a] for a in UNS if a in r and r[a] == r[a]) >= 55]
    print(f"{name}: {len(g)} satir, {len({r['_u'] for r in g})} birim")
    for a, b in [('A3_topp', 'A2_hbody'), ('A7_resid', 'A2_hbody'), ('A3_topp', 'A1_random'), ('A7_resid', 'A3_topp')]:
        v = np.array([r[a] - r[b] for r in g]); lo, hi = boot(g, lambda r: r[a] - r[b])
        print(f"  {a}-{b}: medyan {np.median(v):+.2f} [{lo:+.2f},{hi:+.2f}]  pozitif {np.sum(v > 0)}/{len(v)}")
