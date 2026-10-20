"""Gurultulu secim deneyi -- analiz. Kullanim: python nsel_analyze.py <out_klasoru> <t11_json_klasoru>"""
import sys, os, glob, json, pickle, math
import numpy as np

ROOT = os.environ.get('HOPE_ROOT', os.path.expanduser('~/Documents/hope-research'))
CUT = "# ============================================================================\n# SELF-CHECK"
ns = dict(np=np, xp=np, GPU=False, CDT=np.complex128, FDT=np.float64,
          to_xp=lambda a, dt=None: np.asarray(a, dtype=dt) if dt else np.asarray(a),
          to_np=np.asarray, free_mem=lambda: None)
exec(open(os.path.join(ROOT, 'src', 'gate4_core.py'), encoding='utf-8').read().split(CUT)[0], ns)
s_star = ns['s_star']
S_GRID = [1e2, 3e2, 1e3, 3e3, 1e4, 1e5, np.inf]
S_SEL = [1e2, 3e2, 1e3, 3e3]

out, jdir = sys.argv[1], sys.argv[2]
rows = []
for f in ('t11v3_FULL_results.json', 't11fmv3_FULL_results.json'):
    rows += [r for r in json.load(open(os.path.join(jdir, f)))['rows']
             if r.get('live') and r.get('ec') == 'EC-LEARN' and r.get('k6_ok')]
R = {}
for f in glob.glob(os.path.join(out, '*.pkl')):
    d = pickle.load(open(f, 'rb'))
    R[d['unit']['key']] = d
rows = [r for r in rows if r['key'] in R]
print(f'{len(rows)} satir, {len(set(r["key"] for r in rows))} birim tamam')


def curve(r, arm, ssel):
    a = R[r['key']]['acc']
    return [float(np.mean(a[(r['H'], r['p'], arm, ssel, S)])) for S in S_GRID]


def S_of(r, arm, ssel):
    return s_star(S_GRID, curve(r, arm, ssel), r['tau'])


def ratio(sh, s_kernel, sel_cost):
    den = s_kernel + sel_cost
    return 0.0 if not math.isfinite(den) else sh / den


def boot(vals_by_unit, B=4000, seed=0):
    keys = list(vals_by_unit)
    allv = [x for k in keys for x in vals_by_unit[k]]
    rng = np.random.default_rng(seed)
    bs = [np.median([x for k in rng.integers(0, len(keys), len(keys)) for x in vals_by_unit[keys[k]]])
          for _ in range(B)]
    return float(np.median(allv)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def show(label, fn):
    by = {}
    for r in rows:
        by.setdefault(r['key'], []).append(fn(r))
    m, lo, hi = boot(by)
    print(f'  {label:<58} {m:7.2f}x  [{lo:6.2f}, {hi:6.2f}]')
    return m


# --- 1. dogrulama: gurultusuz secim FULL'u uretiyor mu ---
print('\n1) Dogrulama (secim gurultusuz):')
for arm, col in (('A7', 'A7_resid'), ('A3', 'A3_topp')):
    lr = [math.log10(S_of(r, arm, np.inf) / r[f'S_{col}']) for r in rows
          if math.isfinite(S_of(r, arm, np.inf)) and math.isfinite(r[f'S_{col}'])]
    dacc = [curve(r, arm, np.inf)[-1] - r[f'curve_{col}'][-1] for r in rows]
    print(f'  {arm}: log10(S*_bizim / S*_FULL) medyan {np.median(lr):+.3f}, |.|>0.2: {sum(abs(x) > 0.2 for x in lr)}/{len(lr)}'
          f' | gurultusuz dogruluk farki medyan {np.median(dacc):+.2f} puan, maks |.| {np.max(np.abs(dacc)):.2f}')
show('greedy-residual, secim maliyetsiz (FULL S_h, bizim S*)', lambda r: ratio(r['S_A2_hbody'], S_of(r, 'A7', np.inf), 0))
show('ayni, FULL S* ile (17 Ekim rakami: 39.6x)', lambda r: ratio(r['S_A2_hbody'], r['S_A7_resid'], 0))

# --- 2. asil soru ---
ARMS = (('A7', 'greedy-residual'), ('A7d', 'greedy-residual, yanlilik duzeltilmis'),
        ('A3', 'top-capacity'), ('A3d', 'top-capacity, yanlilik duzeltilmis'))
BASE = {'A7': 'A7', 'A7d': 'A7', 'A3': 'A3', 'A3d': 'A3'}
SUMMARY = {}
for arm, name in ARMS:
    print(f'\n2) {name}: secim S_sel shot/nokta ile, maliyet = S*(S_sel) + f*S_sel')
    for ss in S_SEL:
        SUMMARY[f'{arm}|{int(ss)}'] = show(f'S_sel = {int(ss):>5}', lambda r, ss=ss: ratio(r['S_A2_hbody'], S_of(r, arm, ss),
                                                               R[r['key']]['m_frac'] * ss))
    if arm == 'A7':
        show('17 Ekim varsayimi: gurultusuz secim + f*1000 maliyet',
             lambda r: ratio(r['S_A2_hbody'], S_of(r, 'A7', np.inf), R[r['key']]['m_frac'] * 1000))
        show('17 Ekim varsayimi: kendi S*: x(1+f)',
             lambda r: ratio(r['S_A2_hbody'], S_of(r, 'A7', np.inf) * (1 + R[r['key']]['m_frac']), 0))

# --- 3. secim gurultusu dogrulugu ne kadar bozuyor ---
print('\n3) Gurultusuz kernel dogrulugu (S=inf), secim gurultulu - gurultusuz secim, medyan [min, maks] puan:')
for arm, _ in ARMS:
    parts = []
    for ss in S_SEL:
        d = [curve(r, arm, ss)[-1] - curve(r, BASE[arm], np.inf)[-1] for r in rows]
        parts.append(f'{int(ss)}: {np.median(d):+.2f} [{np.min(d):+.1f}, {np.max(d):+.1f}]')
    print(f'  {arm:<4}', ' | '.join(parts))
print('\n4) Hedefe hic varamayan satir (S*=inf) sayisi:')
for arm, _ in ARMS:
    lv = ([np.inf] if arm in ('A7', 'A3') else []) + S_SEL
    print(f'  {arm:<4}', ' '.join(f'{("inf" if np.isinf(ss) else int(ss))}:{sum(1 for r in rows if not math.isfinite(S_of(r, arm, ss)))}' for ss in lv))

# --- 5. kalici results.json ---
if len(sys.argv) > 3:
    def fin(x):
        return None if not math.isfinite(x) else float(x)
    J = dict(
        description='Gurultulu secim deneyi: kapasite/artik kapasite secimi sonlu shot ile olculen Pauli '
                    'degerlerinden yapilinca hedef dogruluga varis icin gereken shot (S*) ve secim dahil shot orani.',
        date='2026-10-17', author="the authors",
        source=dict(hope_research_commit='88babd2', full_results=['t11v3_FULL_results.json', 't11fmv3_FULL_results.json'],
                    row_filter="live & ec=='EC-LEARN' & k6_ok (n=8-12)", scripts=['nsel_unit.py', 'nsel_analyze.py', 'runner.py']),
        params=dict(S_GRID=[None if np.isinf(x) else x for x in S_GRID], S_SEL=S_SEL, n_split=30, seed0=20260916,
                    noise_model='E_sel = E_tr + N(0, 3^w / S_sel), terimler ve noktalar arasi bagimsiz (gate4_core.noisy_features ile ayni)',
                    kernel='RAW: gurultulu ozelliklerin kosinusu; S=inf gurultusuz', target='tau = H-body gurultusuz dogruluk - 1.0 (FULL satirindan)',
                    cost='oran = S*_hbody / (S*_kol(S_sel) + f * S_sel), f = egitim payi (~0.7); S_sel=inf -> secim maliyetsiz',
                    arms={'A7': 'greedy-residual (phaseb_core.select_greedy_residual)', 'A7d': 'greedy-residual, puan - (M-k)/M * 3^w/S_sel',
                          'A3': 'top-capacity', 'A3d': 'top-capacity, kapasite - 3^w/S_sel'},
                    seeds='crc32(birim|bolme|H|p): secim gurultusu; +1: kernel gurultusu'),
        env='Colab A100-SXM4-40GB, cupy 14.0.1, scikit-learn 1.6.1, float32',
        summary=dict(median_ratio=SUMMARY),
        rows=[])
    for r in rows:
        a = R[r['key']]['acc']
        e = dict(key=r['key'], ds=r['ds'], n=r['n'], ent=r['ent'], bw=r['bw'], H=r['H'], p=r['p'],
                 tau=r['tau'], m_frac=R[r['key']]['m_frac'], S_hbody_full=fin(r['S_A2_hbody']),
                 S_A7_full=fin(r['S_A7_resid']), S_A3_full=fin(r['S_A3_topp']), hbody_ref=r.get('hbody_ref'),
                 arms={})
        for arm, _ in ARMS:
            for ss in ([np.inf] if arm in ('A7', 'A3') else []) + S_SEL:
                tag = f'{arm}|{"inf" if np.isinf(ss) else int(ss)}'
                per = [a[(r['H'], r['p'], arm, ss, S)] for S in S_GRID]
                Sst = S_of(r, arm, ss)
                cost = 0 if np.isinf(ss) else R[r['key']]['m_frac'] * ss
                e['arms'][tag] = dict(curve=curve(r, arm, ss), acc_per_split=per, S_star=fin(Sst),
                                      ratio_vs_hbody=ratio(r['S_A2_hbody'], Sst, cost))
        e['picks_split0'] = {str(k): v for k, v in R[r['key']]['picks'].get((r['H'], r['p']), {}).items()}
        J['rows'].append(e)
    json.dump(J, open(sys.argv[3], 'w'), ensure_ascii=False)
    print(f'\nresults.json yazildi: {sys.argv[3]} ({len(J["rows"])} satir)')
