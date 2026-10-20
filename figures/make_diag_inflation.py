"""Diagonal inflation measurement (Section 4.3).

Bir sentetik ayarda (ZZ, depth 8, eps 1, bw 0.25, n = 8/10/12, M = 400 etiketsiz girdi, havuz w <= 2, I yok)
her kural icin: (a) tutulan terimlerin iz payi sum_S gamma_i (psi olcegi, tum 4^n terimin toplami 1),
(b) normalize kernelin etkin rank orani effrank/M (5.1'deki 'kimlige yakin' olcutu, 0.9 esigi),
(c) normalize kernelin kosegen disi standart sapmasi, (d) normalize oncesi kosegen-disi std / kosegen ortalamasi.
Kaynak kod: hope-research src/qk_core.py, src/phaseb_core.py (exec ile, CPU numpy).
Calistir: python make_diag_inflation.py /path/to/hope-research  (CPU, ~10 s)
"""
import json, sys, os, numpy as np
xp = np; GPU = False; CDT = np.complex128; FDT = np.float64
to_xp = lambda a, dt=None: np.asarray(a, dtype=dt) if dt is not None else np.asarray(a)
to_np = lambda a: np.asarray(a)
HR = sys.argv[1]
exec(open(os.path.join(HR, 'src/qk_core.py')).read())
exec(open(os.path.join(HR, 'src/phaseb_core.py')).read())

def stats(F, scale):
    """F: (M, |S|) Pauli beklenen degerleri <P>; scale = 2^n (psi = <P>/sqrt(2^n))."""
    K = F @ F.T / scale                     # normalize edilmemis sum_S psi_i psi_i'
    dg = np.diag(K).copy(); M = len(K); off = ~np.eye(M, dtype=bool)
    Kn = K / np.sqrt(np.outer(dg, dg))
    ev = np.linalg.eigvalsh(Kn)
    return dict(trace_share=float(dg.mean()), effrank_frac=float(effrank(ev) / M),
                offdiag_std_norm=float(Kn[off].std()), offdiag_mean_norm=float(Kn[off].mean()),
                ratio_unnorm=float(K[off].std() / dg.mean()))

out = dict(setting=dict(depth=8, ent=1.0, bw=0.25, M=400, w_max=2, seed=0), rows=[])
for n in (8, 10, 12):
    rng = np.random.default_rng(1000 + n)
    X = sample_X(rng, 400, n)
    psi = encode(X, n, 8, 1.0, 0.25)
    Kf = np.abs(psi.conj() @ psi.T) ** 2
    M = len(Kf); off = ~np.eye(M, dtype=bool)
    out['rows'].append(dict(n=n, rule='full', p=4 ** n, trace_share=1.0,
                            effrank_frac=float(effrank(np.linalg.eigvalsh(Kf)) / M),
                            offdiag_std_norm=float(Kf[off].std()), offdiag_mean_norm=float(Kf[off].mean()),
                            ratio_unnorm=float(Kf[off].std())))
    lay = pool_layout(n, 2)
    _, _, idx, w = lay
    E = pauli_expect_fast(psi, n, lay)          # (M, L) <P>
    keep = w >= 1                                # I (agirlik 0) havuzda yok
    E, w = E[:, keep], w[keep]
    gam = capacity(E)                            # <P>^2 olcegi
    rows = [('pool (all w<=2)', np.arange(E.shape[1]), None)]
    for p in (16, 64):
        rows.append(('top-capacity', select_topp(gam, p), p))
        rows.append(('greedy-residual', np.asarray(select_greedy_residual(E, p)), p))
        for h in (1, 2):
            for k, s in enumerate(select_hbody_exact(w, h, p, rng, 5)):
                rows.append((f'H-body H={h}', s, p))
        for s in select_random(E.shape[1], p, rng, 5):
            rows.append(('random', np.asarray(s), p))
    for rule, S, p in rows:
        st = stats(E[:, S], 2 ** n)
        st.update(n=n, rule=rule, p=int(p) if p else int(E.shape[1]),
                  cap_mean_psi=float(gam[S].mean() / 2 ** n), weight_mean=float(w[S].mean()))
        out['rows'].append(st)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'diag_inflation_results.json'), 'w'), indent=1)

import collections
agg = collections.defaultdict(list)
for r in out['rows']: agg[(r['n'], r['rule'], r['p'])].append(r)
print(f"{'n':>3} {'rule':<16} {'p':>6} {'iz payi':>9} {'effrank/M':>9} {'std(Kn)':>8} {'mean(Kn)':>8} {'std/diag':>9}")
for (n, rule, p), v in agg.items():
    m = lambda k: np.mean([x[k] for x in v])
    print(f"{n:>3} {rule:<16} {p:>6} {m('trace_share'):>9.4f} {m('effrank_frac'):>9.3f} {m('offdiag_std_norm'):>8.3f} {m('offdiag_mean_norm'):>8.3f} {m('ratio_unnorm'):>9.4f}")
