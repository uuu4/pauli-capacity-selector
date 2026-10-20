"""KAPI 4 cekirdegi -- KAYNAK MUHASEBESI.  TEK KAYNAK.

qk_core : devre, Pauli domeni, kernel, EC teshisi (Katman A/B/C)
phaseb  : etiketler, kol secimi (A1-A7), merge, KRR, istatistik, Katman D
gate4   : shot muhasebesi -- shadow varyansi, gurultulu kernel, S*, butce

BEKLENEN ORTAM: qk_core + phaseb_core'un tanimladigi her sey +
`np/xp/GPU/CDT/FDT/to_xp/to_np` (defterin 0. hucresi).

================================ PARA BIRIMI =================================
1 shot = 1 DEVRE CALISTIRMASI. Iki olcum protokolu AYNI birimde sayilir ama
kernel matrisine FARKLI carpanla donusur:

  shadow   (A1-A8): S snapshot x NOKTA sayisi. Tum (x,y) ciftleri
                    post-processing'den bedava gelir  ->  maliyet M ile LINEER.
  fidelity (A0)   : compute-uncompute, her CIFT icin ayri devre
                                                     ->  maliyet M^2 ile.

Bu `M` carpani Kapi 4'un en buyuk tek kalemidir ve hata #17'nin ("A0'in mutlak
shot sayilari GFQK ile ayni para birimi degil") cozumudur: birim ayni, carpan
farkli, ikisi de acikca yazili.

Ikinci eksen (devre derinligi) `twoq_gates` ile ayri raporlanir; ucuncu eksen
(model karmasikligi `p`) kollar arasi kiyasta SABIT tutulur. Protokol geregi
uc eksen BIRLIKTE degil AYRI AYRI raporlanir (01_TEST_PROTOCOL.md, KAPI 4).
==============================================================================
"""

SHADOW_BASE = 3.0


def op_cost(w):
    """`dP_k` -- bir Pauli'nin classical-shadow olcum maliyeti: `3^agirlik`.

    A8'in maliyet vektoru budur (`select_greedy_residual(..., cost=op_cost(w))`,
    yani HOPE'un tam `argmin J_k/dP_k` kurali). Kapi 1-3'te `dP_k = 1` alinmisti;
    protokol A8'i "shot muhasebesi Kapi 4'un konusu" diye burada aciyor.
    """
    return SHADOW_BASE ** np.asarray(w, dtype=float)


def dir_var(w, C=None, model='l1'):
    """Yonlerin TEK SNAPSHOT varyansi. Doner `(p,)`.

    `C=None`   -> ciplak Pauli'ler: `v_k = 3^{w_k}`.
    `C` verili -> birlestirilmis yon `f_k = sum_i C_ik P_i` (A4):
      'l1' (BIRINCIL, muhafazakar): shadow normu alt-toplamsal,
           `Var <= (sum_i |C_ik| 3^{w_i/2})^2`
      'l2' (IYIMSER alt sinir): bilesen tahminleri BAGIMSIZmis gibi,
           `Var = sum_i C_ik^2 3^{w_i}`

    Ikisi de hesaplanip raporlanir; `l1/l2` orani merge'in SHOT EKSENINDEKI
    bedelidir (olcut G2). Protokolun 2. sorusu -- "bilesen Pauli'leri ayri ayri
    mi gerekiyor?" -- cevabi: shadow altinda EK SNAPSHOT gerekmez (ayni anlik
    goruntuler tum Pauli'leri besler), ama birlesik yonun VARYANSI buyuyebilir.
    Yani merge'in maliyeti "sifir" degil: "ekstra devre yok, ekstra varyans var".
    """
    c = op_cost(w)
    if C is None:
        return c
    A = np.abs(np.asarray(C, dtype=float))
    if model == 'l1':
        return (A.T @ np.sqrt(c)) ** 2
    if model == 'l2':
        return (A.T ** 2) @ c
    raise ValueError(model)


def sigma2_arm(F, v):
    """`S=1` shot varyansi, birim-kosegene normalize kernel icin. `qk_core.
    kernel_gfqk` ile AYNI form: `2 * sum_k v_k <f_k^2> / <diag>^2`.

    F: `(M, p)` yon degerleri · v: `(p,)` tek-snapshot varyanslari.
    `S` shot'ta varyans `sigma2/S`, dolayisiyla `S_req = sigma2 / Var[k~]`
    (`1/Var` DEGIL -- hata #15).
    """
    mf2 = to_np(xp.mean(F ** 2, 0)).astype(float)
    dg = float(to_np(xp.mean(xp.sum(F ** 2, 1))))
    return float(2.0 * np.sum(np.asarray(v, dtype=float) * mf2) / max(dg, 1e-30) ** 2)


def noisy_features(F, v, S, rng):
    """`f_hat = f + N(0, v/S)`. Doner `(F_hat, diag_hat)`; diag YANLILIK DUZELTILMIS.

    `E[sum_k f_hat_k^2] = sum_k f_k^2 + sum_k v_k/S` -> duzeltilmezse dusuk
    normlu noktalar yapay olarak birbirine benzer gorunur (hata #16 ile ayni
    aile: gurultu ekseni ters donerdi). Cikarilir, pozitife kirpilir.

    BILINEN BASITLESTIRME (on-kayitli): ayni shadow kumesi TUM operatorleri
    besledigi icin `eps`'ler gercekte KORELELIDIR; burada bagimsiz alinir. Bu
    hata TUM kollarda ayni yonde -> `S*` ORANLARI icin zararsiz, MUTLAK `S*`
    icin iyimser. Kapi 4'un karari oranlar uzerinden verilir.
    """
    sd = np.sqrt(np.asarray(v, dtype=float) / float(S))
    G = F + to_xp(rng.standard_normal(size=tuple(F.shape)) * sd[None, :], FDT)
    return G, xp.maximum(xp.sum(G ** 2, 1) - float(np.sum(sd ** 2)), 1e-30)


def noisy_fidelity(K, S, rng, symmetric=False):
    """A0'in gurultu modeli: compute-uncompute BINOM tahmin edici.

    `k_hat ~ Binom(S, k)/S` -> `Var = k(1-k)/S`. Shadow'daki `3^w` yuku YOK;
    A0'in bedeli varyansta degil, CIFT SAYISINDA (bkz. `budget`).

    `symmetric=True`: egitim blogu icin. Girdi bloklari bagimsiz gurultulendigi
    icin `K_hat` simetrik cikmaz; `krr_multi` `eigh` kullandigindan simetrize
    edilir. Bu ayni zamanda fiziksel olarak da dogru: `(x,y)` ve `(y,x)` AYNI
    devredir, iki kez olculmez -- yani simetrizasyon shot da kazandirmaz,
    `budget('fidelity', ...)` zaten ust ucgeni sayar.
    """
    Kc = np.clip(to_np(K).astype(float), 0.0, 1.0)
    Kh = rng.binomial(int(S), Kc) / float(int(S))
    if symmetric:
        Kh = (Kh + Kh.T) / 2.0
    return to_xp(Kh, FDT)


def s_star(S_grid, accs, target, interp=True):
    """Hedefi saglayan ve DAHA BUYUK her `S`'te de saglayan en kucuk `S`.

    Monotonluk sarti, tek bir gurultu dalgalanmasinin `S*`'i yapay olarak
    kucultmesini engeller (kol karsilastirmasinda bu dogrudan sahte kazanc
    olurdu). `S_grid` ARTAN olmali. Hicbiri saglamiyorsa `inf`.

    `interp=True` (varsayilan): kesisme `log10(S)`'te DOGRUSAL ara degerle
    bulunur. GEREKCE: grid adimi yarim dekad (~3.16x); grid'e yapistirilmis
    `S*` yalnizca `{1, 3.16, 10, ...}` oranlarini uretebilir, yani "2x tasarruf"
    gibi bir esik OLCULEMEZ hale gelir. Ara deger cozunurlugu surekli yapar.
    Gridin ILK noktasinda zaten saglaniyorsa `S_grid[0]` doner (yani "<= S_min",
    ust sinir olarak okunur); `inf` girdisinde ara deger yapilmaz.
    """
    a = list(accs)
    for i, S in enumerate(S_grid):
        if not all(x >= target for x in a[i:]):
            continue
        if i == 0 or not interp or np.isinf(S):
            return float(S)
        s0, a0, a1 = S_grid[i - 1], a[i - 1], a[i]      # a0 < target <= a1
        if a1 <= a0:
            return float(S)
        f = (target - a0) / (a1 - a0)
        return float(10.0 ** (np.log10(s0) + f * (np.log10(S) - np.log10(s0))))
    return float('inf')


def budget(protocol, S, m_tr, m_te, m_sel=0):
    """Tum kernel matrisini kurmak icin TOPLAM devre calistirmasi.

    `m_sel` = kriter (`gamma`) icin olculen SECIM seti. A1/A2 kriter kullanmaz
    -> `m_sel=0`; A3/A4/A7/A8 kullanir. Ayni anlik goruntuler havuzdaki TUM
    `M_H` operatoru besledigi icin ek maliyet operator sayisinda LOGARITMIK
    (bkz. 01_TEST_PROTOCOL.md 2.5) -- yani `m_sel` NOKTA sayisiyla dogrusal,
    havuz buyuklugunden bagimsiz sayilir. Kriter maliyetinin karari cevirip
    cevirmedigi G5'te ayrica raporlanir.
    """
    if protocol == 'shadow':
        return float(S) * (m_tr + m_te + m_sel)
    if protocol == 'fidelity':
        return float(S) * (m_tr * (m_tr - 1) / 2.0 + m_te * m_tr)
    raise ValueError(protocol)


def twoq_gates(n, depth, ent, protocol):
    """Devre basina 2-qubit kapi sayisi (ikinci maliyet ekseni).

    ZZ-feature map: katman basina `n(n-1)/2` ZZ dondurmesi, `ent=0` ise hic.
    Fidelity compute-uncompute -> 2 kat. Shadow'un rastgele olcum katmani TEK
    QUBIT'tir, 2q sayisina katki YOK.
    """
    g = depth * (n * (n - 1) // 2) if ent > 0 else 0
    return 2 * g if protocol == 'fidelity' else g


def weight_mix(w, idx):
    """Secimin ortalama agirligi + `3^w` cinsinden ortalama maliyeti.

    G3'un teshisi: A8'in dusuk agirliga COKUP cokmedigi (yani A2'ye yakinsayip
    yakinsamadigi) buradan gorulur -- `select_greedy_residual` docstring'inde
    on-kayitli olarak beklenen sonuc budur.
    """
    ww = np.asarray(w, dtype=float)[np.asarray(idx, dtype=int)]
    return float(ww.mean()), float(op_cost(ww).mean())


# ============================================================================
# SELF-CHECK -- `python src/gate4_core.py`
# ============================================================================

def _selftest():
    rng = np.random.default_rng(0)
    M, n, H = 400, 6, 3

    # [1] ciplak secimde l1 == l2 == 3^w  (C=I ozel hali)
    w = np.array([0, 1, 2, 3, 1])
    assert np.allclose(dir_var(w), 3.0 ** w)
    I = np.eye(len(w))
    assert np.allclose(dir_var(w, I, 'l1'), 3.0 ** w)
    assert np.allclose(dir_var(w, I, 'l2'), 3.0 ** w)
    print('[1] OK  C=I -> l1 == l2 == 3^w')

    # [2] l1 >= l2 her zaman (Cauchy-Schwarz); merge bedeli negatif OLAMAZ
    C = rng.normal(size=(len(w), 3))
    assert np.all(dir_var(w, C, 'l1') >= dir_var(w, C, 'l2') - 1e-9)
    print('[2] OK  l1 >= l2 (merge bedeli >= 0)')

    # [3] sigma2_arm == kernel_gfqk'nin s2'si (gurultusuz, tam w<=H havuzu)
    #     Iki bagimsiz yol ayni sayiyi vermeli -> muhasebe qk_core ile TUTARLI.
    X = rng.uniform(0, 2 * np.pi, size=(M, n))
    lay = pool_layout(n, H)
    E = pauli_expect_fast(encode(X, n, 2, 1.0, 0.3), n, lay)
    wp = np.asarray(lay[3])
    _, s2_ref = kernel_gfqk(E, wp, H, 0.0)
    m = wp <= H
    s2_new = sigma2_arm(E[:, to_xp(m)], op_cost(wp[m]))
    assert abs(s2_new - s2_ref) / s2_ref < 1e-5, (s2_new, s2_ref)
    print(f'[3] OK  sigma2_arm == kernel_gfqk.s2  ({s2_new:.4g})')

    # [4] kosegen yanliligi cikiyor: cok cekimde <diag_hat> ~ diag
    idx = np.argsort(to_np(xp.mean(E ** 2, 0)))[::-1][:16]
    F = E[:, to_xp(np.asarray(idx))]
    v = dir_var(wp[idx])
    d0 = to_np(xp.sum(F ** 2, 1))
    acc = np.zeros_like(d0)
    for _ in range(200):
        acc += to_np(noisy_features(F, v, 5e4, rng)[1])
    rel = float(np.abs(acc / 200 - d0).mean() / d0.mean())
    assert rel < 0.02, rel
    print(f'[4] OK  diag yanliligi duzeltildi (bagil sapma {rel:.2e})')

    # [5] gurultu S ile 1/S olceklenir ve sigma2_arm'i tutturur
    #     -> ANALITIK S_req ile AMPIRIK S* ayni para biriminde (G4'un temeli)
    Kc = to_np(kernel_pair(F, F))
    off = ~np.eye(M, dtype=bool)
    for S in (1e3, 1e4):
        d = []
        for _ in range(30):
            G, dg = noisy_features(F, v, S, rng)
            d.append(to_np(kernel_pair(G, G, dg, dg))[off] - Kc[off])
        emp = float(np.var(np.concatenate(d)))
        pred = sigma2_arm(F, v) / S
        assert 0.5 < emp / pred < 2.0, (S, emp, pred)
        print(f'[5] OK  S={S:.0e}  ampirik/analitik varyans = {emp / pred:.2f}')

    # [5b] A0'in BINOM modeli: Var == k(1-k)/S, simetrizasyon calisiyor
    Kf, _ = kernel_fidelity(encode(X[:120], n, 2, 1.0, 0.3))
    for S in (1e3, 1e4):
        d = [to_np(noisy_fidelity(Kf, S, rng)) - to_np(Kf) for _ in range(40)]
        K0 = to_np(Kf)
        pred = float(np.mean(K0 * (1 - K0)) / S)
        emp = float(np.var(np.concatenate([x.ravel() for x in d])))
        assert 0.5 < emp / pred < 2.0, (S, emp, pred)
    Ks = noisy_fidelity(Kf, 1e3, rng, symmetric=True)
    assert np.allclose(to_np(Ks), to_np(Ks).T)
    print('[5b] OK  A0 BINOM varyansi k(1-k)/S, simetrik blok simetrik')

    # [6] S* monotonluk sarti: gecici bir sicrama S*'i kucultmemeli
    grid = [10, 100, 1000, 10000]
    assert s_star(grid, [60, 40, 40, 60], 55, interp=False) == 10000.0
    assert s_star(grid, [40, 40, 60, 60], 55, interp=False) == 1000.0
    assert s_star(grid, [40, 40, 40, 40], 55) == float('inf')
    assert s_star(grid, [60, 60, 60, 60], 55) == 10.0          # ilk noktada saglandi
    # ara deger: 100 -> 40, 1000 -> 60, hedef 55  =>  f=0.75 => 10^2.75
    s = s_star(grid, [40, 40, 60, 60], 55)
    assert abs(s - 10 ** 2.75) < 1e-6, s
    assert s_star([10, 100, np.inf], [40, 40, 60], 55) == float('inf')  # inf'te ara deger yok
    print(f'[6] OK  s_star monotonluk + log-ara deger ({s:.0f} vs grid 1000)')

    # [7] para birimi: fidelity/shadow orani M mertebesinde (Kapi 4'un ana kalemi)
    #     (M_tr(M_tr-1)/2 + M_te*M_tr) / (M_tr+M_te) = 399800/1200 ~ 333
    bs = budget('shadow', 1, 400, 800)
    bf = budget('fidelity', 1, 400, 800)
    assert 300 < bf / bs < 400, bf / bs
    assert budget('shadow', 1, 400, 800, m_sel=1000) == bs + 1000
    assert twoq_gates(8, 8, 1.0, 'fidelity') == 2 * twoq_gates(8, 8, 1.0, 'shadow')
    assert twoq_gates(8, 8, 0.0, 'shadow') == 0
    print(f'[7] OK  fidelity/shadow butce orani = {bf / bs:.0f}x  (M~1200)')

    # [8] A8 gercekten maliyet-farkinda: pahali ama guclu yonu ATLAR
    Eo = np.linalg.qr(rng.normal(size=(M, 4)))[0] * np.sqrt(M)
    Eo = to_xp(Eo * np.array([3.0, 2.9, 2.8, 2.7]), FDT)
    w4 = np.array([4, 1, 1, 1])                 # 0. sutun en guclu AMA en pahali
    a7 = select_greedy_residual(Eo, 2)
    a8 = select_greedy_residual(Eo, 2, cost=op_cost(w4))
    assert 0 in set(map(int, a7)), a7
    assert 0 not in set(map(int, a8)), a8
    print(f'[8] OK  A7 pahali yonu aliyor {list(map(int, a7))}, '
          f'A8 atliyor {list(map(int, a8))}')

    print('\nKAPI 4 cekirdegi: 8/8 GECTI')


if __name__ == '__main__':
    import os
    import numpy as np                                    # noqa: F811
    _h = os.path.dirname(os.path.abspath(__file__))
    G = dict(__name__='gate4_core', np=np, xp=np, GPU=False,
             CDT=np.complex128, FDT=np.float64,
             to_xp=lambda a, dtype=None: np.asarray(a, dtype=dtype),
             to_np=np.asarray)
    for _f in ('qk_core.py', 'phaseb_core.py', 'gate4_core.py'):
        exec(compile(open(os.path.join(_h, _f), encoding='utf-8').read(), _f, 'exec'), G)
    G['_selftest']()
