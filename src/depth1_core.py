"""depth=1 kapali form -- agirlik <= 2 Pauli beklenen degerleri ve A0, statevector'suz ya da ucuz.

`encode(X, n, depth=1, ent, bw)` = H^n|0> sonra kosegen faz:
    phi(s) = -( sum_i a_i s_i + sum_{i<j} b_ij s_i s_j ),  a_i = bw x_i,  b_ij = ent bw^2 x_i x_j
F = flip (X/Y) qubitleri, G = faz (Y/Z) qubitleri. `pauli_expect_fast` konvansiyonuyla
    <P> = Re[ (-i)^{|F&G|} 2^{-|F|} sum_{s_F} prod_{q in F} s_q^[q in G] e^{2i a_q s_q} prod_{j notin F} phi_j ]
    phi_j = cos(2 t_j) (j notin G) | i sin(2 t_j) (j in G),   2 t_j = sum_{q in F} 2 b_qj s_q
F bos ise (yalniz Z) <P> = 0: genlik buyuklukleri duzgun. Maliyet terim basina O(n), 2^n yok.

A0 (fidelity) icin kapali form YOK (tum ciftli faz toplami, IQP). Ama faz yalniz
T(z) = sum_i x_i s_i(z)'ye bagli: sum_{i<j} x_i x_j s_i s_j = (T^2 - |x|^2)/2. Boylece
2^n x C(n,2) isaret tablosu kurulmaz; durum O(n 2^n) ile uretilir (n<=26 A100'de).

BEKLENEN ORTAM: `xp/GPU/CDT/FDT/to_xp/to_np` + qk_core (`diagnose`, `kernel_fidelity`).
Yalniz `norm='none'` (T1.1 ve FM kosularinin ayari).
"""


def pool2_terms(n):
    """Agirlik 1-2 havuzu, I YOK (v2'den beri hicbir kolun havuzunda degil).

    Doner (qa, la, qb, lb, w): harf 1=X 2=Y 3=Z (`pool_layout` pidx rakamlariyla ayni);
    agirlik-1 terimde qb = -1. Sira: once agirlik 1, sonra (i<k) ciftleri.
    """
    qa, la, qb, lb = [], [], [], []
    for i in range(n):
        for L in (1, 2, 3):
            qa.append(i); la.append(L); qb.append(-1); lb.append(0)
    for i in range(n):
        for k in range(i + 1, n):
            for Li in (1, 2, 3):
                for Lk in (1, 2, 3):
                    qa.append(i); la.append(Li); qb.append(k); lb.append(Lk)
    qa, la, qb, lb = map(np.asarray, (qa, la, qb, lb))
    return qa, la, qb, lb, np.where(qb < 0, 1, 2)


def pool2_pidx(terms, n):
    """`pool_layout` pidx'i (yalniz kucuk n'de dogrulama icin): qubit q -> rakam 4^{n-1-q}."""
    qa, la, qb, lb, _ = terms
    pid = la.astype(object) * (4 ** (n - 1 - qa)).astype(object)
    m = qb >= 0
    pid[m] = pid[m] + lb[m].astype(object) * np.asarray([4 ** (n - 1 - int(q)) for q in qb[m]], dtype=object)
    return pid


def expect_depth1(X, n, ent, bw, terms=None):
    """(M, K) beklenen degerler, sira `pool2_terms(n)` ile ayni. float64 (98 cos carpimi)."""
    qa, la, qb, lb, _ = pool2_terms(n) if terms is None else terms
    D = xp.float64
    Xd = to_xp(X, D)
    M = Xd.shape[0]
    A = 2.0 * bw * Xd                                          # 2 a_q
    B = 2.0 * ent * bw**2 * Xd[:, :, None] * Xd[:, None, :]    # 2 b_qj
    eye = xp.eye(n, dtype=bool)[None]
    C = xp.where(eye, 1.0, xp.cos(B))
    S = xp.where(eye, 0.0, xp.sin(B))
    # Pex[:, q, r] = prod_{j notin {q, r}} C[:, q, j]  (ozel carpim, bolme yok)
    one = xp.ones((M, n, 1), dtype=D)
    pre = xp.cumprod(xp.concatenate([one, C[:, :, :-1]], 2), 2)
    suf = xp.flip(xp.cumprod(xp.concatenate([one, xp.flip(C, 2)[:, :, :-1]], 2), 2), 2)
    Pex = pre * suf
    Pi = xp.prod(C, 2)                                         # prod_{j != q} C[:, q, j]

    E = xp.zeros((M, len(qa)), dtype=D)
    # --- agirlik 1: X_q = cos(2a) Pi, Y_q = sin(2a) Pi, Z_q = 0
    for L, f in ((1, xp.cos), (2, xp.sin)):
        c = np.where((qb < 0) & (la == L))[0]
        E[:, to_xp(c)] = f(A[:, qa[c]]) * Pi[:, qa[c]]
    # --- agirlik 2, tek flip q, digeri r = Z:
    #     X_q Z_r = -sin(2a_q) sin(2b_qr) Pex_qr,  Y_q Z_r = cos(2a_q) sin(2b_qr) Pex_qr
    w2 = qb >= 0
    for Lq, sgn, f in ((1, -1.0, xp.sin), (2, 1.0, xp.cos)):
        for side in (0, 1):                                    # flip a'da mi b'de mi
            if side == 0:
                c = np.where(w2 & (la == Lq) & (lb == 3))[0]; q, r = qa[c], qb[c]
            else:
                c = np.where(w2 & (lb == Lq) & (la == 3))[0]; q, r = qb[c], qa[c]
            qx, rx = to_xp(q), to_xp(r)
            E[:, to_xp(c)] = sgn * f(A[:, qx]) * S[:, qx, rx] * Pex[:, qx, rx]
    # --- agirlik 2, iki flip (i, k): Q_sigma = prod_{j notin {i,k}} cos(2b_ij + sigma 2b_kj)
    c2 = np.where(w2 & (la != 3) & (lb != 3))[0]
    for i in range(n - 1):
        ks = np.arange(i + 1, n)
        Bi, Bk = B[:, i, None, :], B[:, ks, :]                 # (M,1,n), (M,nk,n)
        mask = np.ones((len(ks), n), dtype=bool)
        mask[:, i] = False
        mask[np.arange(len(ks)), ks] = False
        mk = to_xp(mask)[None]
        Qp = xp.prod(xp.where(mk, xp.cos(Bi + Bk), 1.0), 2)    # (M, nk)  s_i = s_k
        Qm = xp.prod(xp.where(mk, xp.cos(Bi - Bk), 1.0), 2)    #          s_i = -s_k
        sel = c2[qa[c2] == i]
        kk = qb[sel] - (i + 1)
        gi, gk = (la[sel] == 2), (lb[sel] == 2)                # Y -> G'de
        acc = xp.zeros((M, len(sel)), dtype=xp.complex128)
        Ai, Ak = A[:, i][:, None], A[:, to_xp(qb[sel])]
        for si in (1.0, -1.0):
            for sk in (1.0, -1.0):
                Q = (Qp if si == sk else Qm)[:, to_xp(kk)]
                amp = to_xp(np.where(gi, si, 1.0) * np.where(gk, sk, 1.0))
                acc += amp[None] * xp.exp(1j * (Ai * si + Ak * sk)) * Q
        ph = to_xp((-1j) ** (gi.astype(int) + gk.astype(int)))
        E[:, to_xp(sel)] = xp.real(ph[None] * acc) / 4.0
    return E


def states_depth1(X, n, ent, bw):
    """(B, 2^n) durumlar, `encode(.., depth=1, ..)` ile ayni (global faz haric)."""
    Xd = to_xp(X, xp.float64)
    T = xp.zeros((Xd.shape[0], 1), dtype=xp.float64)
    for i in range(n):                                         # qubit 0 = en anlamli bit
        T = xp.stack([T + Xd[:, i:i+1], T - Xd[:, i:i+1]], -1).reshape(Xd.shape[0], -1)
    ph = -(bw * T + 0.5 * ent * bw**2 * T * T)
    return (xp.exp(1j * ph) * 2.0 ** (-n / 2)).astype(CDT)


def fidelity_depth1(Xa, Xb, n, ent, bw, block=16):
    """|<psi(a)|psi(b)>|^2, bloklu (n=24'te durum 128 MB, n=26'da 512 MB). Doner np (Ma, Mb)."""
    K = np.zeros((len(Xa), len(Xb)))
    for s in range(0, len(Xa), block):
        Pa = states_depth1(Xa[s:s+block], n, ent, bw)
        for t in range(0, len(Xb), block):
            Pb = states_depth1(Xb[t:t+block], n, ent, bw)
            K[s:s+block, t:t+block] = to_np(xp.abs(Pa.conj() @ Pb.T) ** 2)
            del Pb
        del Pa
        if GPU:
            xp.get_default_memory_pool().free_all_blocks()
    return K


# ============================================================================
# SELF-CHECK  (defterde Bolum 1 ayni testi yapar)
# ============================================================================
if __name__ == '__main__':
    import os
    import numpy as np
    xp, GPU = np, False
    CDT, FDT = np.complex128, np.float64
    to_xp = lambda a, dtype=None: np.asarray(a, dtype=dtype)
    to_np = np.asarray
    _H = os.path.dirname(os.path.abspath(__file__))
    for fn in ('qk_core.py',):
        exec(open(os.path.join(_H, fn), encoding='utf-8').read().split('# SELF-CHECK')[0])
    rng = np.random.default_rng(0)
    for n in (6, 8, 10):
        for ent, bw in ((1.0, 0.15), (0.5, 0.25), (1.0, 0.6)):
            X = rng.uniform(0, 2 * np.pi, size=(7, n))
            lay = pool_layout(n, 2)
            Esv = pauli_expect_fast(encode(X, n, 1, ent, bw), n, lay)[:, np.asarray(lay[3]) >= 1]
            pid_sv = np.asarray(lay[2])[np.asarray(lay[3]) >= 1]
            tm = pool2_terms(n)
            pid = pool2_pidx(tm, n)
            o = np.argsort(pid.astype(float))
            assert np.array_equal(pid[o].astype(np.int64), pid_sv.astype(np.int64)), 'havuz farkli'
            Ecf = expect_depth1(X, n, ent, bw, tm)[:, o]
            err = np.abs(Ecf - Esv).max()
            Kf = np.abs(encode(X, n, 1, ent, bw).conj() @ encode(X, n, 1, ent, bw).T) ** 2
            errK = np.abs(fidelity_depth1(X, X, n, ent, bw, block=3) - Kf).max()
            print(f'n={n} ent={ent} bw={bw}: |E_cf - E_sv| = {err:.1e}   |K0_T - K0_sv| = {errK:.1e}')
            assert err < 1e-12 and errK < 1e-12
    print('OK')
