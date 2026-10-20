"""Gurultulu secim deneyi -- tek birim.

Soru: secim (kapasite / artik kapasite) gercekte sonlu shot'la olculen Pauli degerlerinden
yapilirsa, hedef dogruluga varis icin gereken shot (S*) ne kadar bozulur?

Boru hatti T11 v3 defteriyle (build_t11_nb.py, hope-research@88babd2) ayni: bolme, on-isleme,
havuz (H<=4, max_flips_for), RAW gurultulu kernel, krr_multi; GPU varsa cupy + float32 (defter gibi).
Tek fark: secim girdisi E_sel = E_tr + N(0, 3^w / S_sel)  (gate4_core.noisy_features ile ayni model).
S_sel = inf -> defterin kendisi (dogrulama).

Kullanim: python nsel_unit.py <birim_json> <cikti_klasoru>   (cwd: data/ alt klasoru olan dizin)
NSPLIT ortam degiskeni bolme sayisini kisaltir (deneme icin).
"""
import sys, os, json, pickle, time, zlib
import numpy as np

ROOT = os.environ.get('HOPE_ROOT', os.path.expanduser('~/Documents/hope-research'))
CUT = "# ============================================================================\n# SELF-CHECK"
try:
    import cupy as cp
    cp.cuda.runtime.getDeviceCount()
    xp, GPU = cp, True
except Exception:
    cp, xp, GPU = None, np, False
CDT = xp.complex64 if GPU else xp.complex128
FDT = xp.float32 if GPU else xp.float64


def to_xp(a, dtype=None):
    return xp.asarray(a, dtype=dtype)


def to_np(a):
    return cp.asnumpy(a) if GPU and isinstance(a, cp.ndarray) else np.asarray(a)


def free_mem():
    if GPU:
        cp.get_default_memory_pool().free_all_blocks()


ns = dict(np=np, xp=xp, GPU=GPU, CDT=CDT, FDT=FDT, to_xp=to_xp, to_np=to_np, free_mem=free_mem)
for f in ('qk_core.py', 'phaseb_core.py', 'gate4_core.py', 't11_core.py'):
    exec(open(os.path.join(ROOT, 'src', f), encoding='utf-8').read().split(CUT)[0], ns)
g = ns.get

SEED0 = 20260916
S_GRID = [1e2, 3e2, 1e3, 3e3, 1e4, 1e5, np.inf]
S_SEL = [np.inf, 1e2, 3e2, 1e3, 3e3]


def max_flips_for(n):
    return None if n >= 16 else {8: 200, 10: 120, 12: 120, 14: 60}.get(n, 200)


def batch_for(n):
    return int(min(512, max(8, 2 ** 25 >> n)))


def feats(Xs, n, depth, ent, bw, layout):
    b = batch_for(n)
    return xp.concatenate([g('pauli_expect_fast')(g('encode')(Xs[i:i + b], n, depth, ent, bw=bw, norm='none'),
                                                  n, layout) for i in range(0, len(Xs), b)], axis=0)


def greedy_debiased(E, p, vS, block=4096):
    """Gurultu yanliligi duzeltilmis greedy-residual: phaseb_core.select_greedy_residual ile ayni
    Gram-Schmidt, ama secim puani artik kapasiteden beklenen gurultu payi cikarilarak verilir.
    Bagimsiz gurultu (varyans vS_i her girdide) k yon izdusuldukten sonra artikta ~(M-k)/M * vS_i
    birakir; puan = ||r_i||^2/M - (M-k)/M * vS_i. Rank dolgusu duzeltilmis kapasiteyle."""
    R = xp.array(E, copy=True)
    M, K = R.shape
    v = to_xp(np.asarray(vS, dtype=np.float64), FDT)
    nrm = xp.sum(R ** 2, axis=0) / M
    chosen = []
    for k in range(min(p, K)):
        score = nrm - v * ((M - k) / M)
        if chosen:
            score[to_xp(np.asarray(chosen))] = -xp.inf
        j = int(to_np(xp.argmax(score)))
        if float(to_np(nrm[j])) <= 1e-24 or float(to_np(score[j])) <= 0:
            break
        chosen.append(j)
        q = R[:, j] / xp.sqrt(xp.sum(R[:, j] ** 2))
        for s in range(0, K, block):
            sl = slice(s, min(s + block, K))
            R[:, sl] -= xp.outer(q, q @ R[:, sl])
        nrm = xp.sum(R ** 2, axis=0) / M
    if len(chosen) < min(p, K):
        cap = to_np(xp.sum(E ** 2, axis=0) / M - v).astype(float)
        cap[np.asarray(chosen, dtype=int)] = -np.inf
        chosen += [int(j) for j in np.argsort(-cap, kind='stable')[:min(p, K) - len(chosen)]]
    return np.asarray(chosen)


def main(unit_path, out_dir):
    U = json.load(open(unit_path))
    ds, n, depth, ent, bw, key = U['ds'], U['n'], U['depth'], U['ent'], U['bw'], U['key']
    HP = [tuple(x) for x in U['hp']]
    X, y = g('load_real')(ds)
    splits = g('stratified_splits')(y, 30, SEED0, 0.7)[:int(os.environ.get('NSPLIT', 30))]
    layout = g('pool_layout')(n, 4, max_flips=max_flips_for(n), seed=SEED0, keep_low=2)
    w_pool = np.asarray(to_np(layout[3]))
    acc, picks, m_frac = {}, {}, []
    t0 = time.time()
    for si, (tr, te) in enumerate(splits):
        Xt, Xe = g('preprocess')(X[tr], X[te], n, seed=SEED0)
        yt, ye = y[tr], y[te]
        m_frac.append(len(tr) / float(len(tr) + len(te)))
        Et = feats(Xt, n, depth, ent, bw, layout)
        Ee = feats(Xe, n, depth, ent, bw, layout)
        for (H, p) in HP:
            mi = np.where((w_pool <= H) & (w_pool >= 1))[0]
            ixH = to_xp(mi)
            EtH, EeH, wH = Et[:, ixH], Ee[:, ixH], w_pool[mi]
            vH = np.asarray(to_np(g('dir_var')(wH)), dtype=float)
            seed = zlib.crc32(f'{key}|{si}|{H}|{p}'.encode())
            srng = np.random.default_rng(seed)
            nrng = np.random.default_rng(seed + 1)
            sels = {}
            for Ss in S_SEL:
                if np.isinf(Ss):
                    Es = EtH
                else:
                    Es = EtH + to_xp(srng.standard_normal(tuple(EtH.shape)) * np.sqrt(vH / Ss)[None, :], FDT)
                gam = np.asarray(to_np(g('capacity')(Es)), dtype=float)
                sels[('A3', Ss)] = np.asarray(to_np(g('select_topp')(gam, p)))
                if not np.isinf(Ss):
                    sels[('A3d', Ss)] = np.asarray(to_np(g('select_topp')(gam - vH / Ss, p)))
                sels[('A7', Ss)] = np.asarray(to_np(g('select_greedy_residual')(Es, p)))
                if not np.isinf(Ss):
                    sels[('A7d', Ss)] = greedy_debiased(Es, p, vH / Ss)
                if not np.isinf(Ss):
                    del Es
            free_mem()
            uniq = {}
            for k, idx in sels.items():
                uniq.setdefault(tuple(sorted(int(i) for i in idx)), []).append(k)
            if si == 0:
                picks[(H, p)] = {str(k): [int(i) for i in idx] for k, idx in sels.items()}
            ilist = list(uniq.keys())
            for S in S_GRID:
                Kts, Kes = [], []
                for ii in ilist:
                    ix = to_xp(np.asarray(ii))
                    Ft, Fe = EtH[:, ix], EeH[:, ix]
                    if np.isinf(S):
                        G, Ge = Ft, Fe
                    else:
                        v1 = g('dir_var')(wH[np.asarray(ii)])
                        G, _ = g('noisy_features')(Ft, v1, S, nrng)
                        Ge, _ = g('noisy_features')(Fe, v1, S, nrng)
                    D3 = xp.maximum(xp.sum(G ** 2, 1), 1e-30)
                    D4 = xp.maximum(xp.sum(Ge ** 2, 1), 1e-30)
                    Kts.append(G @ G.T / xp.sqrt(D3[:, None] * D3[None, :]))
                    Kes.append(Ge @ G.T / xp.sqrt(D4[:, None] * D3[None, :]))
                R = g('krr_multi')(xp.stack(Kts), xp.stack(Kes), {'y': yt}, {'y': ye})
                for b, ii in enumerate(ilist):
                    a = R[(b, 'y')][0] if len(ilist) > 1 else R['y'][0]
                    for k in uniq[ii]:
                        acc.setdefault((H, p) + k + (S,), []).append(float(a))
                del Kts, Kes, R
                free_mem()
            del EtH, EeH
        del Et, Ee
        free_mem()
        if si == 0:
            print(f'{key}: bolme 0 {time.time() - t0:.1f} sn (GPU={GPU})', flush=True)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, key + '.pkl'), 'wb') as fh:
        pickle.dump(dict(unit=U, acc=acc, picks=picks, m_frac=float(np.mean(m_frac)),
                         secs=time.time() - t0), fh)
    print(f'{key}: BITTI {time.time() - t0:.0f} sn', flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
