"""Faz D v3 doz analizi (makale Deney 4): dozlara gore greedy-residual - top-capacity farki, yuksek-dusuk doz farki
ve fark-rho korelasyonu, bootstrap 4000. Calistir (repo kokunden): python runs/phase_dv3_full_v1/d23.py
"""
import json, numpy as np
d = json.load(open('runs/phase_dv3_full_v1/phase_dv3_FULL_results.json'))
dec = [r for r in d['decisions'] if r.get('rho_A3') is not None]
DOSES = [(1, 0.0), (2, 0.5), (2, 0.25), (2, 0.0)]
N = 4000
def corr_ci(x, y, seed=0):
    x, y = np.asarray(x, float), np.asarray(y, float); r0 = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed); bs = []
    while len(bs) < N:
        i = rng.integers(0, len(x), len(x))
        if np.std(x[i]) > 0 and np.std(y[i]) > 0: bs.append(np.corrcoef(x[i], y[i])[0, 1])
    return r0, np.percentile(bs, 2.5), np.percentile(bs, 97.5)
def md_ci(a, b, seed=0):
    a, b = np.asarray(a, float), np.asarray(b, float); pt = np.median(a) - np.median(b)
    rng = np.random.default_rng(seed)
    st = [np.median(a[rng.integers(0, len(a), len(a))]) - np.median(b[rng.integers(0, len(b), len(b))]) for _ in range(N)]
    return pt, np.percentile(st, 2.5), np.percentile(st, 97.5)
for name, filt in [('qlocal DAHIL (protokol)', lambda r: True), ('qlocal HARIC', lambda r: r['label'] != 'qlocal')]:
    rs = [r for r in dec if r['comparison'] == 'A7 resid vs A3 top-p' and filt(r)]
    seen = [z for z in DOSES if any((r['rep'], r['jitter']) == z for r in rs)]
    lo = [r['diff'] for r in rs if (r['rep'], r['jitter']) == seen[0]]
    hi = [r['diff'] for r in rs if (r['rep'], r['jitter']) == seen[-1]]
    meds = {z: (len([r for r in rs if (r['rep'], r['jitter']) == z]), np.median([r['diff'] for r in rs if (r['rep'], r['jitter']) == z])) for z in seen}
    pt, l, h = md_ci(hi, lo); r0, rl, rh = corr_ci([r['rho_A3'] for r in rs], [r['diff'] for r in rs])
    print(f'{name}: n={len(rs)}  doz medyanlari ' + ', '.join(f'{z}: n={n} {m:+.2f}' for z, (n, m) in meds.items()))
    print(f'   D2 fark (yuksek-dusuk) {pt:+.2f} [{l:+.2f}, {h:+.2f}]   D3 r={r0:+.3f} [{rl:+.3f}, {rh:+.3f}]')
    for arm_cmp in ['A3 top-p vs A2 H-body', 'A7 resid vs A2 H-body']:
        rr = [r for r in dec if r['comparison'] == arm_cmp and filt(r)]
        if rr: print(f'   {arm_cmp}: ' + ', '.join(f'{z}: {np.median([r["diff"] for r in rr if (r["rep"], r["jitter"]) == z]):+.2f}' for z in seen))
print(sorted({r['comparison'] for r in dec}))
