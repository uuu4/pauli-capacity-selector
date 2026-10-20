"""Kuantum kernel cekirdegi -- TEK KAYNAK.

Bu dosya defterlere OLDUGU GIBI GOMULUR (build_*_nb.py `open(...).read()` ile alir),
boylece Colab'de self-contained kalir ama kod tek yerde durur.

BEKLENEN ORTAM: `xp`, `GPU`, `CDT`, `FDT`, `to_xp`, `to_np` defterin 0. hucresinde
tanimlanmis olmali (cupy/numpy dagitimi).

Dogrulanmis (bkz. defterlerin Bolum 4):
  fwht == H^{otimes n} · pauli_expect_fast == naif yol (2.2e-16) · Parseval ·
  GFQK_{w<=n} == fidelity kernel · dort eksende S_req monoton · Katman B/C ayrimlari
"""


# ============================================================================
# DEVRE -- yogun 2^n x 2^n matris KURULMAZ
# ============================================================================

def fwht(a, normalize=True):
    """Son eksende H^{otimes n}. a: (..., 2^n). O(N n 2^n), yogun matris yok.

    YERINDE kelebek (butterfly). Onceki surum her adimda `moveaxis`+`stack`
    yapiyordu; profil bunun tek basina Faz B suresinin ~%15'i oldugunu gosterdi
    (7600 `stack` cagrisi). H tek-qubit kapilari komutatif oldugu icin bit
    sirasindan bagimsiz; dogrulama [1a] H^{otimes n}'e karsi kilitliyor.
    """
    dim = a.shape[-1]
    n = int(np.log2(dim))
    shp = a.shape
    b = xp.array(a, copy=True).reshape(-1, dim)
    h = 1
    while h < dim:
        b = b.reshape(-1, dim // (2*h), 2, h)
        u = xp.array(b[:, :, 0, :], copy=True)
        v = b[:, :, 1, :]
        b[:, :, 0, :] = u + v
        b[:, :, 1, :] = u - v
        b = b.reshape(-1, dim)
        h *= 2
    b = b.reshape(shp)
    return b * (2.0 ** (-n / 2)) if normalize else b


def sample_X(rng, N, n, rep=1, jitter=0.0):
    """Veri ureteci -- FAZ D icin REDUNDANS KOLU.

    `rep=1, jitter=0.0` -> eski yolun BIREBIR aynisi (ayni tek `rng` cagrisi,
    dolayisiyla Faz A/B/C/C2 sonuclari degismez).

    `rep>1`: n koordinat, `d_eff = max(3, n//rep)` GIZLI degiskenden turetilir:

        x_j = z_{j mod d_eff} + jitter * u_j,    z~U(0,2pi),  u~U(-pi,pi)

    Bu redundansi VERIDE uretir, DEVREDE degil. Amac kritik: devrenin ifade gucu
    (EC'nin kaynagi) dokunulmadan kalirken `psi` sutunlari korele hale gelsin.
    Devreyi degistirseydik EC'yi de degistirir ve testi anlamsizlastirirdik.

    `d_eff >= 3` tutulur cunku etiket bataryasi ilk 3 koordinati kullanir
    (`make_labels`, `n_fixed=3`) -- gorevin icsel zorlugu sabit kalmali.

    UYARI (raporlanmali): `rep>1` veri manifoldunun icsel boyutunu dusurur, yani
    ogrenme gorevini kolaylastirabilir. Bu, A7-vs-A3 karsilastirmasi icin ZARARSIZ
    (iki kol da ayni veriyi gorur) ama mutlak dogruluklar Faz C2 ile KIYASLANAMAZ.
    """
    if rep == 1 and jitter == 0.0:
        return rng.uniform(0, 2 * np.pi, size=(N, n))      # BIREBIR eski yol
    d = max(3, n // rep)
    z = rng.uniform(0, 2 * np.pi, size=(N, d))
    X = z[:, np.arange(n) % d]
    if jitter > 0.0:
        X = X + jitter * rng.uniform(-np.pi, np.pi, size=(N, n))
    return X


def sign_tables(n):
    z = np.arange(2 ** n)
    S = 1.0 - 2.0 * ((z[:, None] >> np.arange(n - 1, -1, -1)[None, :]) & 1)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    SP = np.stack([S[:, i] * S[:, j] for i, j in pairs], 1) if pairs else np.zeros((2 ** n, 0))
    return S, SP, pairs


_SIGN_DEV = {}


def sign_tables_dev(n):
    """`sign_tables`'in cihazdaki (FDT, transpoze) kopyasi, SON `n` icin onbellekli.

    2026-10-13 (kosu 2, n=16/20): eski `encode` tabloyu HER cagrida yeniden kurup
    cihaza tasiyordu; n=20'de bu cagri basina ~1.5 GB (SP: 2^20 x 190). Degerler
    birebir ayni -- yalniz bir kez kurulur. Tek `n` tutulur: n=20 tablosu ~0.8 GB.
    """
    if n not in _SIGN_DEV:
        _SIGN_DEV.clear()
        S, SP, pairs = sign_tables(n)
        _SIGN_DEV[n] = (to_xp(S, FDT).T, to_xp(SP, FDT).T, pairs)
    return _SIGN_DEV[n]


def encode(X, n, depth, ent, bw, norm='none'):
    """ZZ-feature map, `depth` tekrar. Doner: (M, 2^n) statevector.

    faz(z) = -( bw * sum_i x_i s_i(z)  +  ent * bw^2 * sum_{i<j} x_i x_j s_ij(z) )

    `norm` -- `n`-BAGIMLI KONFOUND kontrolu:
      'none' : ham. Terim sayilari n ve n(n-1)/2 oldugu icin faz sacilimi
               sirasiyla ~sqrt(n) ve ~n buyur -> BUYUK n'de devre DAHA COK karisir.
               Bu, "n buyudu" ile "encoding guclendi" etkilerini karistirir.
      'sqrt' : her iki terim kendi terim sayisinin karekokune bolunur -> faz
               sacilimi n'den BAGIMSIZ. "Ayni harita, farkli n" yorumu gecerli olur.

    ent=0 -> faz tamamen tek-qubit -> durum CARPIM kalir (Meyer-Wallach tam 0),
    `norm` bunu degistirmez (tek-qubit terim carpimsal yapiyi bozmaz).
    """
    St, SPt, pairs = sign_tables_dev(n)
    Xp = np.stack([X[:, i] * X[:, j] for i, j in pairs], 1) if pairs else np.zeros((len(X), 0))
    c1, c2 = 1.0, 1.0
    if norm == 'sqrt':
        c1 = 1.0 / np.sqrt(n)
        c2 = 1.0 / np.sqrt(max(len(pairs), 1))
    elif norm != 'none':
        raise ValueError(norm)
    phase = -(c1 * bw * (to_xp(X, FDT) @ St)
              + c2 * ent * (bw ** 2) * (to_xp(Xp, FDT) @ SPt))
    D = xp.exp(1j * phase).astype(CDT)
    psi = xp.zeros((len(X), 2 ** n), dtype=CDT)
    psi[:, 0] = 1.0
    for _ in range(depth):
        psi = fwht(psi) * D
    return psi


def meyer_wallach(psi, n):
    """Q = 2(1 - (1/n) sum_k tr(rho_k^2)).  0 = carpim, 1 = maksimal dolanik."""
    M = psi.shape[0]
    tot = xp.zeros(M, dtype=FDT)
    for k in range(n):
        a = xp.moveaxis(psi.reshape(M, *([2] * n)), k + 1, 1).reshape(M, 2, -1)
        rho = xp.einsum('mir,mjr->mij', a, a.conj())
        tot += xp.real(xp.einsum('mij,mji->m', rho, rho)).astype(FDT)
    return float(to_np(xp.mean(2.0 * (1.0 - tot / n))))


# ============================================================================
# PAULI DOMENI -- flip-maskesi + Walsh-Hadamard hizli yolu
# ============================================================================

def _popcount(v):
    v = np.asarray(v, dtype=np.int64).copy()
    r = np.zeros_like(v)
    while np.any(v):
        r += v & 1
        v >>= 1
    return r


def _spread4(v, n):
    v = np.asarray(v, dtype=np.int64)
    out = np.zeros_like(v)
    for k in range(n):
        out += ((v >> k) & 1) * (4 ** k)
    return out


def pool_layout_from_indices(idx, n):
    """Verilen KEYFI Pauli indeks kumesi icin layout (agirlik havuzu degil).
    `qglobal` etiketi gibi tek tuk operator gerektiren yerlerde kullanilir."""
    idx = np.unique(np.asarray(idx, dtype=np.int64))
    f_ = np.zeros(len(idx), np.int64); g_ = np.zeros(len(idx), np.int64)
    for k in range(n):
        d = (idx // (4 ** k)) % 4                  # 0=I 1=X 2=Y 3=Z
        f_ |= np.where((d == 1) | (d == 2), 1 << k, 0)
        g_ |= np.where((d == 2) | (d == 3), 1 << k, 0)
    blocks, fs = [], []
    for f in np.unique(f_):
        m = f_ == f
        blocks.append((g_[m], idx[m])); fs.append(f)
    w = np.zeros(len(idx), np.int64)
    for k in range(n):
        w += (((idx // (4 ** k)) % 4) != 0)
    o = np.argsort(idx)
    return np.array(fs), blocks, idx[o], w[o]


def pool_layout(n, h, max_flips=None, seed=0, keep_low=None):
    """Agirlik <= h havuzu.

    `max_flips`: flip-maske sayisi bunu asarsa RASTGELE alt orneklenir. Maliyet
    flip-maske sayisiyla DOGRU ORANTILI (her biri bir WHT), operator sayisiyla degil;
    o yuzden kisitlama burada yapilir. Alt ornekleme TUM kollara ayni havuzu verir,
    yani adildir -- ve A3'u (top-p) A1'den (rastgele) DAHA COK zorlar, cunku en iyi
    operator disarida kalabilir. Bizim iddiamiz acisindan MUHAFAZAKAR.

    `keep_low` (v2, 2026-10-15 denetimi): popcount <= keep_low olan flip'lerin HEPSI
    tutulur, `max_flips` yalniz geri kalanlardan (popcount > keep_low) cekilir. Neden:
    H=2 kollari H=4 icin kirpilmis havuzdan cekiliyordu (n=12'de 2-body 154/594, ZZ
    asiri temsil) ve n=14, depth=1, ent=0'da H=2 havuzunun rank'i 8'e dusup Faz B'yi
    dusurdu. `None` -> eski davranis (eski kosular birebir yeniden uretilir).

    Doner: (flip_maskeleri, [(g_dizisi, pauli_indeksleri) her f icin],
            tum_indeksler_sirali, tum_agirliklar_sirali).
    """
    allz = np.arange(2 ** n)
    pc = _popcount(allz)
    fs = np.where(pc <= h)[0]
    if keep_low is not None:
        lo, hi = fs[pc[fs] <= keep_low], fs[pc[fs] > keep_low]
        if max_flips is not None and len(hi) > max_flips:
            hi = hi[np.random.default_rng(seed + 4242).choice(len(hi), max_flips, replace=False)]
        fs = np.sort(np.concatenate([lo, hi]))
    elif max_flips is not None and len(fs) > max_flips:
        keep = np.random.default_rng(seed + 4242).choice(len(fs), max_flips, replace=False)
        fs = np.sort(fs[np.union1d(keep, 0)])      # f=0 (Z-only aile) HER ZAMAN kalir
    blocks, idx_all, w_all = [], [], []
    for f in fs:
        g = np.where(pc[allz | f] <= h)[0]            # agirlik = |f OR g|
        pidx = 2 * _spread4(f & g, n) + _spread4(f & ~g, n) + 3 * _spread4(g & ~f, n)
        blocks.append((g, pidx))
        idx_all.append(pidx)
        w_all.append(pc[g | f])
    idx_all = np.concatenate(idx_all)
    w_all = np.concatenate(w_all)
    o = np.argsort(idx_all)
    return fs, blocks, idx_all[o], w_all[o]


def pauli_expect_fast(psi, n, layout):
    """<P_{f,g}> = Re[ (-i)^{|f&g|} * WHT_z( conj(psi_z) psi_{z^f} )[g] ]

    Ayni f'yi paylasan tum operatorler tek gather + tek WHT ile hesaplanir.
    n=12, w<=4: 46 666 operator yerine 794 donusum.
    """
    fs, blocks, _, _ = layout
    z = xp.arange(2 ** n)
    conj = psi.conj()
    cols, keys = [], []
    for f, (g, pidx) in zip(fs, blocks):
        Wt = fwht(conj * psi[:, z ^ int(f)], normalize=False)
        ph = ((-1j) ** to_xp(_popcount(np.bitwise_and(int(f), g)))).astype(CDT)
        cols.append(xp.real(Wt[:, to_xp(g.astype(np.int64))] * ph[None, :]).astype(FDT))
        keys.append(pidx)
    E = xp.concatenate(cols, axis=1)
    return E[:, to_xp(np.argsort(np.concatenate(keys)))]


def pauli_expect_naive(psi, n, idx):
    """Referans yol -- SADECE dogrulama icin (yavas)."""
    flip = np.zeros(len(idx), np.int64)
    phm = np.zeros(len(idx), np.int64)
    nY = np.zeros(len(idx), np.int64)
    for k in range(n):
        d = (np.asarray(idx) // (4 ** k)) % 4
        flip |= np.where((d == 1) | (d == 2), 1 << k, 0)
        phm |= np.where((d == 2) | (d == 3), 1 << k, 0)
        nY += (d == 2)
    z = np.arange(2 ** n, dtype=np.int64)
    P = to_np(psi)
    out = np.zeros((P.shape[0], len(idx)))
    for t in range(len(idx)):
        zf = z ^ flip[t]
        sgn = 1.0 - 2.0 * (_popcount(zf & phm[t]) & 1)
        out[:, t] = np.real(np.sum(P.conj() * sgn[None, :] * P[:, zf], 1) * (1j) ** int(nY[t]))
    return out


# ============================================================================
# KERNEL -- birim kosegene normalize + tahmin edicinin shot varyansi
# ============================================================================

def kernel_gfqk(E_pool, w_pool, H, p):
    """GFQK_{w<=H}, gurultulu, birim kosegene normalize. Doner (K~, sigma2_shot).

    Gurultu: qubit basina depolarizing, Pauli-transfer ozdegeri lam=1-4p/3 -> EGZAKT.
    Shot varyansi: classical shadow, operator basina 3^w.
      k = sum_i a_i b_i, Var(a^_i) ~ 3^{w_i}
      -> Var(k^) ~ 2 sum_i 3^{w_i} <psi~_i^2>,   <psi~_i^2> = mean(E[:,i]^2)
    `c`'nin BIRINCI kuvveti kullanilir; c^2 yazmak pay/paydada sadelesip gurultu
    eksenini ters cevirir (bkz. 02_PROJECT_STATUS.md hata #16).
    """
    w = np.asarray(w_pool)
    m = to_xp(w <= H)
    ww = to_xp(w[w <= H].astype(np.float64), FDT)
    E = E_pool[:, m] * xp.sqrt((1.0 - 4.0 * p / 3.0) ** (2 * ww))[None, :]
    K = E @ E.T
    dg = xp.maximum(xp.diag(K), 1e-30)
    s2 = float(to_np(2.0 * xp.sum((3.0 ** ww) * xp.mean(E ** 2, 0)) / xp.mean(dg) ** 2))
    return K / xp.sqrt(xp.outer(dg, dg)), s2


def kernel_fidelity(psi):
    """A0 referansi. diag zaten 1; tahmin edici compute-uncompute BINOM -> sigma2 None.
    Bu, GFQK'dan FARKLI bir olcum protokoludur; mutlak shot sayilari ayni para birimi
    degil (bkz. hata #17)."""
    return xp.abs(psi.conj() @ psi.T) ** 2, None


# ============================================================================
# TESHIS -- Katman A-shot / A-learn / B / C
# ============================================================================

def effrank(ev):
    ev = xp.clip(ev.real, 0, None)
    s1, s2 = ev.sum(), (ev ** 2).sum()
    return float(to_np(s1 * s1 / s2)) if float(to_np(s2)) > 0 else 0.0


def diagnose(K, sigma2_shot=None, budgets=(1e3, 1e5, 1e7)):
    """S_req = sigma2_shot / Var[k~]  --  `1/Var` DEGIL (hata #15)."""
    M = K.shape[0]
    off = K[~xp.eye(M, dtype=bool)]
    var = float(to_np(off.var()))
    mu = float(to_np(off.mean()))
    kc = xp.clip(off, 0.0, 1.0)
    s2 = float(to_np(xp.mean(kc * (1 - kc)))) if sigma2_shot is None else float(sigma2_shot)
    er = effrank(xp.linalg.eigvalsh(K))
    d = dict(offdiag_var=var, offdiag_mean=mu, sigma2_shot=s2, M=M,
             s_req=(s2 / var) if var > 0 else float('inf'),
             effrank_K=er, effrank_frac=er / M)
    for S in budgets:
        d[f'resolv@{S:.0e}'] = float(to_np(xp.mean(xp.abs(off - mu) > 3 * np.sqrt(s2 / S))))
    return d


def layers_ac(d, budget, mean_gate, effrank_gate):
    """Doner (durum, a_shot, a_learn). Katman C ONCE: K->11^T ise bandwidth cokusu, EC DEGIL.

    UYARI: a_learn (effrank/M -> 1) EC'ye OZGU DEGILDIR -- carpim durumlarinda da
    tetiklenir ve n>=14'te 1.0000'da doyar. Bolge listesi Katman B (ustel onayi) ile
    KAPILANMALIDIR (bkz. 02_PROJECT_STATUS.md 6.96).
    """
    if d['offdiag_mean'] >= mean_gate:
        return 'COLLAPSE', False, False
    a_shot = d['s_req'] > budget
    a_learn = d['effrank_frac'] > effrank_gate
    if a_shot and a_learn:
        return 'EC-BOTH', True, True
    if a_shot:
        return 'EC-SHOT', True, False
    if a_learn:
        return 'EC-LEARN', False, True
    return 'NO-EC', False, False


def layer_b(ns, vals, n_boot=2000, seed=0):
    """log V ~ a + b*n (ustel)  vs  log V ~ a' + b'*log n (polinom), AIC ile.
    `exponential` = egim GA ust siniri < 0  VE  AIC ustelde daha kucuk.

    UYARI: `exponential=False`, "EC yok" DEMEK DEGILDIR -- egri henuz bukulme
    bolgesinde olabilir (asimptota girmemis). Bkz. 6.96 acik #1.
    """
    ns = np.asarray(ns, float)
    v = np.log(np.maximum(np.asarray(vals, float), 1e-300))
    k = len(ns)
    if k < 3:
        return None

    def fit(x, y):
        b, a = np.polyfit(x, y, 1)
        r = y - (a + b * x)
        ss = ((y - y.mean()) ** 2).sum()
        return b, (1 - (r ** 2).sum() / ss if ss > 0 else 0.0), (r ** 2).sum()

    b_e, r2_e, ss_e = fit(ns, v)
    b_p, r2_p, ss_p = fit(np.log(ns), v)
    aic = lambda ss: k * np.log(max(ss, 1e-300) / k) + 4
    rng = np.random.default_rng(seed)
    bs = []
    while len(bs) < n_boot:                       # dejenere cekilis (tek n) egimi bozar
        i = rng.integers(0, k, k)
        if len(np.unique(ns[i])) >= 2:
            bs.append(np.polyfit(ns[i], v[i], 1)[0])
    bs = np.array(bs)
    return dict(slope=float(b_e), slope_lo=float(np.percentile(bs, 2.5)),
                slope_hi=float(np.percentile(bs, 97.5)), r2_exp=float(r2_e),
                r2_poly=float(r2_p), d_aic=float(aic(ss_e) - aic(ss_p)),
                exponential=bool(np.percentile(bs, 97.5) < 0 and aic(ss_e) < aic(ss_p)))
