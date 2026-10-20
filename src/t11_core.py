"""T1.1 cekirdegi -- GERCEK VERI (Kapi 6, dis gecerlilik).  TEK KAYNAK.

qk_core : devre, Pauli domeni, kernel, EC teshisi (Katman A/B/C)
phaseb  : etiketler, kol secimi (A1-A8), merge, KRR, istatistik, Katman D
gate4   : shot muhasebesi -- shadow varyansi, gurultulu kernel, S*, butce
t11     : gercek veri yukleme + on-isleme + M-bagimsiz EC kapisi + Huang `g`

BEKLENEN ORTAM: qk_core + phaseb_core + gate4_core'un tanimladigi her sey +
`np/xp/GPU/CDT/FDT/to_xp/to_np` (defterin 0. hucresi).

On-kayit: `01_TEST_PROTOCOL.md` 6.42 (K1-K13) + kilit sonrasi ek D1.
Buradaki her fonksiyon o bolumun bir maddesini uygular; madde numarasi
docstring'de yazilidir. Kilitten sonra kural degistirilmez -- kod kurala uyar.

================================ SENTETIKTEN FARK ============================
Sentetik tasarimin uc varsayimi gercek veride cokuyordu (6.42 tablosu):
  her tohumda YENI X   -> ayni havuzun yeniden bolunmesi   (K3)
  sans tam %50         -> egitim cogunluk orani            (K1)
  etiket ekseni        -> her setin tek gercek `y`'si      (K8)
Bu dosya bu ucunun karsiligini kurar; geri kalan boru hatti AYNEN devralinir.
==============================================================================
"""

# ============================================================================
# 1) VERI -- K8
# ============================================================================

DATASETS = ('ionosphere', 'breast_cancer', 'fashion_mnist')

# Fashion-MNIST (2026-10-12, T1.1 sonuclari GORULDUKTEN SONRA eklendi -- on-kayitli degil).
# Huang+ 2021 / Shaydulin & Wild 2021 / Wang-Qi-Wang 2024 cizgisi: elbise vs gomlek,
# egitim CSV'sinden sinif basina 400 (toplam 800, dengeli). Kaynak TEK: Kaggle
# `zalando-research/fashionmnist` -> `fashion-mnist_train.csv`; satir sirasi kaynaga
# bagli oldugu icin OpenML'e dusulmez (ayni tohum farkli ornek secerdi).
FM_CLASSES = (3, 6)      # 3 = Dress (+1), 6 = Shirt (-1)
FM_PER_CLASS = 400
FM_SEED = 20261012       # karar tarihi


def _find_local(name, roots=('/kaggle/input', '.', 'data')):
    """Veri dosyasini diskte ARA -- mount yolu VARSAYILMAZ.

    Kaggle dataset'leri her zaman ayni yere baglanmaz; yol hardcode edilirse
    kosu baska bir oturumda sessizce cokup `fetch_openml`e duser (internet
    kapaliysa oturum olur). `rglob` ile bulunur, bulunamazsa None.
    """
    import os
    pats = (name + '.csv', name + '.data', name.replace('_', '') + '.csv')
    for r in roots:
        if not os.path.isdir(r):
            continue
        for dp, _, fns in os.walk(r):
            for fn in fns:
                if fn.lower() in pats:
                    return os.path.join(dp, fn)
    return None


def load_real(name):
    """Doner `(X, y)`; `X` (N, d) float, `y` (N,) +-1.  K8.

    `breast_cancer` sklearn icinde gelir (indirme YOK). `ionosphere` once diskte
    aranir (Kaggle input), yoksa OpenML'den cekilir. `as_frame=False` -> pandas
    yolu hic acilmaz; bu ortam bagimliligini bilerek dusuk tutuyoruz.

    `y` uzerinde +-1 gosterimi zorunlu: `krr_multi` `sign` ile calisiyor ve
    `make_labels` da +-1 uretiyordu -- boru hattinin geri kalani degismesin.
    """
    if name == 'breast_cancer':
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer()
        X, t = d.data.astype(float), d.target
    elif name == 'ionosphere':
        p = _find_local('ionosphere')
        if p is not None:
            import csv
            with open(p, newline='', encoding='utf-8') as f:
                rows = [r for r in csv.reader(f) if r]
            if not rows[0][-1].strip().lower() in ('g', 'b'):
                rows = rows[1:]                       # baslik satiri
            X = np.array([[float(v) for v in r[:-1]] for r in rows])
            t = np.array([1 if r[-1].strip().lower() == 'g' else 0 for r in rows])
        else:
            from sklearn.datasets import fetch_openml
            d = fetch_openml('ionosphere', version=1, as_frame=False, parser='liac-arff')
            X = np.asarray(d.data, dtype=float)
            t = np.asarray([str(v).strip().lower() in ('g', 'good', '1') for v in d.target])
    elif name == 'fashion_mnist':
        p = _find_local('fashion-mnist_train')
        if p is None:
            raise FileNotFoundError('fashion-mnist_train.csv yok -- Kaggle dataset '
                                    'zalando-research/fashionmnist eklenmeli')
        import csv
        a, b = FM_CLASSES
        rows = {a: [], b: []}
        with open(p, newline='', encoding='utf-8') as f:
            rd = csv.reader(f)
            next(rd)                                       # baslik: label,pixel1..784
            for r in rd:
                c = int(r[0])
                if c in rows:
                    rows[c].append([float(v) for v in r[1:]])
        rng = np.random.default_rng(FM_SEED)
        Xs, ts = [], []
        for c in (a, b):
            A = np.asarray(rows[c])
            Xs.append(A[np.sort(rng.choice(len(A), FM_PER_CLASS, replace=False))])
            ts.append(np.full(FM_PER_CLASS, 1 if c == a else 0))
        X, t = np.vstack(Xs), np.concatenate(ts)
    else:
        raise ValueError(name)
    y = np.where(np.asarray(t).astype(int) == 1, 1.0, -1.0)
    return np.asarray(X, dtype=float), y


def chance_level(y_tr):
    """K1 -- sans = EGITIM kumesindeki cogunluk sinifi orani (yuzde).

    Test kumesinin dagilimina BAKILMAZ: baksaydik esik (`D_LIVE`, `TAU2`) test
    verisinin bir fonksiyonu olurdu ve "sabit tahminci canli sayildi" hatasi
    olcum tarafindan uretilirdi. Dengeli dogruluk secilmedi -- onceki fazlarla
    karsilastirilabilirligi bozar (6.42 K1).
    """
    y = np.asarray(y_tr)
    return 100.0 * max(float(np.mean(y > 0)), float(np.mean(y <= 0)))


def stratified_splits(y, n_splits, seed0, train_frac=0.7):
    """K2 + K3 + K11 -- %70/30 tabakali bolme, `n_splits` tekrar, tohum `seed0+i`.

    Tabakali olmasinin sayisal gerekcesi (6.42 K2): `ionosphere`'de rastgele
    bolmede test cogunluk oraninin std'si ~3.9 puan, `D_LIVE=5` ile AYNI
    buyuklukte -> hicbir sey ogrenmeyen sabit tahminci canli gorunebilirdi.
    Tabakalida fark ~0.07 puan (P1 bunu assert eder).

    Tekrarlar BAGIMSIZ DEGILDIR (ayni havuz) -> bootstrap GA'lari iyimser
    olabilir; makalede acikca yazilir (K3).
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    out = []
    for i in range(n_splits):
        ss = StratifiedShuffleSplit(n_splits=1, train_size=train_frac,
                                    random_state=seed0 + i)
        tr, te = next(ss.split(np.zeros(len(y)), np.asarray(y)))
        out.append((tr, te))
    return out


def preprocess(X_tr, X_te, n, seed=0):
    """K9 -- standardizasyon -> PCA(n) -> kantil -> U(0, 2pi).  SIRA KILITLI.

    Her adim YALNIZ egitim kumesinde fit edilir (P2 tek `fit` cagrisi sayar).

    Neden kantil (6.42 K9): `encode` periyodik degildir (`x_i·x_j` terimi,
    `qk_core.py:99`), yani `bw`'nin anlami girdinin OLCEGINE baglidir. EC-LEARN
    aileleri `X ~ U(0, 2pi)` ile kalibre edildi. Olculen fark: `U(0,2pi)`'de faz
    std'si tek-qubit 1.25 / cift terim 6.96; `N(0,1)`'de 0.70 / 0.32 -- cift
    (dolanikli) terim ~22x kuculuyor. Olcek eslenmezse "gercek veride EC yok"
    sonucu VERIDEN degil OLCEKTEN gelirdi.

    Bilinen sinir: PCA bilesenleri bagimsiz degildir; yalniz MARJINALLER
    eslenir, ortak dagilim degil. Bu yuzden K10 EC kapisini yeniden olcer.

    `ionosphere`'in 2. oznitelii sabittir (varyans 0) -> `StandardScaler` sifira
    bolmez (sklearn sabit sutunu 0'a esler), ama PCA oncesi bunu acikca test
    ediyoruz (P4).
    """
    from sklearn.preprocessing import StandardScaler, QuantileTransformer
    from sklearn.decomposition import PCA
    sc = StandardScaler().fit(X_tr)
    A, B = sc.transform(X_tr), sc.transform(X_te)
    pca = PCA(n_components=n, random_state=seed).fit(A)
    A, B = pca.transform(A), pca.transform(B)
    # n_quantiles <= egitim boyutu; esitken egitim marjinali ~birebir uniform
    qt = QuantileTransformer(n_quantiles=min(len(A), 1000),
                             output_distribution='uniform',
                             subsample=10**9, random_state=seed).fit(A)
    return qt.transform(A) * 2.0 * np.pi, np.clip(qt.transform(B), 0.0, 1.0) * 2.0 * np.pi


# ============================================================================
# 2) EC KAPISI -- K13: `M`-bagimsiz hale getirme
# ============================================================================
# `effrank_frac = effrank/M` esigi (0.90) ASLINDA bir `m2` esigidir ve `M` ile
# olceklenir. Kapali form (turetildi + sayisal dogrulandi, 6.42 K13):
#     effrank_frac = 1 / (1 + (M-1) m2),   m2 = kosegen-disi k^2 ortalamasi
# (fidelity kernel'de diag(K)=1 -> Sum(lam)=M, Sum(lam^2)=M + M(M-1) m2).
# Ayni kernel (`m2 = 3.5e-4`) M=400'de 0.877 (NO-EC), M=245'te 0.921 (EC-LEARN).
# Sabit 0.90 birakilirsa `ionosphere` GEVSEK kapidan gecer -> EC olmayan birimler
# G1'e girer -> BIZIM LEHIMIZE yanlilik. O yuzden `m2` cinsinden sabitlenir.

M_ANCHOR = 400          # kalibrasyon capasi: esigin konuldugu Faz A2 grid'i
FRAC_ANCHOR = 0.90      # `EFFRANK_GATE`, Faz A2'den beri degismedi


def m2_star(anchor_M=M_ANCHOR, anchor_frac=FRAC_ANCHOR):
    """Kalibre edilmis esigin `m2` karsiligi: (1/0.90 - 1)/399 = 2.785e-4."""
    return (1.0 / anchor_frac - 1.0) / (anchor_M - 1.0)


def effrank_gate_for(M):
    """`M` boyutundaki bir kernel icin esdeger `effrank_frac` esigi.

    M=400 -> 0.9000 (sentetik kararlar BIREBIR ayni) · M=398 -> 0.9005 ·
    M=245 -> 0.9364.  Esik gercek veriye BAKILARAK secilmiyor; mevcut
    kalibrasyondan ve cebirden turetiliyor -> on-kayit ihlali DEGIL (K13).
    """
    return 1.0 / (1.0 + (M - 1.0) * m2_star())


def frac_to_m2(frac, M):
    """Ters donusum -- raporlamada `m2`'yi `M`'den bagimsiz gostermek icin."""
    return (1.0 / max(frac, 1e-12) - 1.0) / (M - 1.0)


# ============================================================================
# 2.5) KLASIK REFERANS -- RBF
# ============================================================================
# `runs/g1rerun_full_v1/classical_vs_quantum.py`'nin protokolu devralinir
# (medyan sezgisi x carpan izgarasi), TEK farkla: bant genisligi burada LOO ile
# secilir. Gerekce: D1.1 `K_C`'yi "boru hattindaki RBF, LOO-secili bant
# genisligiyle" diye tanimliyor; test kumesine bakarak secilen bir `gamma` ile
# `g` hesaplamak metrigi test verisine baglardi.
#
# Eski kosuyla kiyaslanabilirlik icin defter AYRICA test-en-iyisini de raporlar
# (`rbf_best_test`): o sayi klasigin LEHINE yanli, yani bizim ALEYHIMIZE --
# muhafazakar taraf.

RBF_MULTS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0)


def rbf_K(A, B, gamma):
    d2 = (A*A).sum(1)[:, None] + (B*B).sum(1)[None, :] - 2.0*A @ B.T
    return np.exp(-gamma * np.maximum(d2, 0.0))


def rbf_gamma_grid(X_tr, mults=RBF_MULTS):
    """Medyan sezgisi: `gamma = mult / medyan(||x-x'||^2)`."""
    Z = X_tr[:200]
    med2 = float(np.median(((Z[:, None] - Z[None, :])**2).sum(-1)))
    return [m / max(med2, 1e-12) for m in mults]


def loo_lambda(K, y, lams=None):
    """`(lam, loo_acc)` -- egzakt LOO ile secilen ridge parametresi.

    NEDEN AYRI BIR FONKSIYON: `phaseb_core.krr_multi` bu secimi zaten yapiyor
    ama `lam`'i DISARI VERMIYOR, yalniz test dogrulugunu donduruyor. D1.1 `s_K`
    icin "o kolun KENDI LOO-secili lambda'si" gerekiyor. `krr_multi`'nin imzasini
    degistirmek Faz A-D'nin ALTI yayimlanmis kosusunun ortak yolunu oynatir --
    on kadar satiri burada tekrar etmek o riskten ucuz. Kapali form AYNI:
        H = U diag(ev/(ev+lam)) U^T,  yhat_i^{-i} = (yhat_i - h_i y_i)/(1 - h_i)
    """
    lams = LAMS if lams is None else lams
    A = np.asarray(to_np(K), dtype=float)
    yv = np.asarray(to_np(y), dtype=float).ravel()
    ev, U = np.linalg.eigh((A + A.T)/2.0)
    ev = np.clip(ev, 0.0, None)
    UtY = U.T @ yv
    U2 = U**2
    best = (None, -1.0)
    for lam in lams:
        f = ev/(ev + lam)
        h = U2 @ f
        loo = (U @ (f*UtY) - h*yv)/np.maximum(1.0 - h, 1e-6)
        a = float(np.mean(np.sign(loo) == yv))
        if a > best[1]:
            best = (lam, a)
    return best


def rbf_select_loo(X_tr, y_tr, mults=RBF_MULTS, lams=None):
    """`(gamma, lam, loo_acc)` -- bant genisligi + ridge, ikisi de LOO ile."""
    best = (None, None, -1.0)
    for g in rbf_gamma_grid(X_tr, mults):
        lam, a = loo_lambda(rbf_K(X_tr, X_tr, g), y_tr, lams)
        if a > best[2]:
            best = (g, lam, a)
    return best


# ============================================================================
# 3) HUANG METRIKLERI -- kilit sonrasi ek D1 (ikincil, BAGLAYICI DEGIL)
# ============================================================================
# Kaynak: Huang+ 2021 (arXiv:2011.01938), Ek F/G. Formuller PDF'ten dogrulandi
# (papers/Huang_PowerOfData_2011.01938.pdf, 2026-09-17).
# UYARI (bilgi notundan): EC altinda K_Q ~ I -> `g` BUYUK cikar ama `s_Q ~ N`
# oldugu icin kuantum model hicbir sey ogrenemez. `g` TEK BASINA OKUNMAZ;
# her zaman `s_Q`, `s_C` ile birlikte raporlanir (D1.2 md.4).

# D2 (2026-09-21): izgara ASAGI uzatildi. Huang'in KURALI
# degismedi ("g_tra<=0.045 saglayan en buyuk lam"), yalniz aday kumesi genisledi.
# Gerekce OLCULDU (hicbir `g` DEGERI gorulmeden, yalniz tanimsizlik gorulerek):
#   ionosphere RBF Gram, iz-normalize: ozdeger_min = 4.9e-07, kosul sayisi 3.2e+08
#   g_tra ~ lam/(ozdeger_min+lam)  ->  sart lam <= 0.045*ozdeger_min ~ 2.2e-08
# Eski izgaranin en kucugu 1e-05, yani gerekenden ~3 mertebe buyuktu -> `g` TUM
# birimlerde tanimsiz cikiyordu (6/6 duman testinde). Bu, verinin degil izgaranin
# ozelligiydi. UYARI: bu kadar kucuk `lam`'da regularize `g` ham `g`'ye yaklasir
# -- regularizasyonun onlemek istedigi seye. Bu yuzden `g` YANINDA daima
# `K_C`'nin en kucuk ozdegeri ve kosul sayisi raporlanir (defter Bolum 8).
LAM_GRID_G = (1e-9, 1e-8, 1e-7, 1e-5, 1e-4, 1e-3, 0.01, 0.025, 0.05, 0.1)
G_TRA_MAX = 0.045       # Huang eq. F20 / L9-L10 -- DEGISMEDI


def trace_normalize(K, N=None):
    """`K <- N K / Tr(K)` -- Huang Thm.1'in normalizasyon sarti (D1.1).

    A7 gibi kosegeni 1 OLMAYAN kollar icin zorunlu; atlanirsa `g` kollar arasi
    kiyaslanabilir olmaz (olcek farki metrige sizar).
    """
    A = np.asarray(to_np(K), dtype=float)
    N = A.shape[0] if N is None else N
    tr = float(np.trace(A))
    return A * (N / tr) if tr > 0 else A


def _psd_sqrt(A):
    """Simetrik PSD karekok -- `scipy.linalg.sqrtm` DEGIL.

    `sqrtm` genel (kompleks olabilen) yol; burada matrisler simetrik PSD ve
    M <= 400, yani `eigh` hem ucuz hem gercek-degerli kaliyor. Negatif ozdegerler
    sayisal artiktir, sifira kirpilir.
    """
    ev, V = np.linalg.eigh((A + A.T) / 2.0)
    return (V * np.sqrt(np.clip(ev, 0.0, None))) @ V.T


def _lmax(A):
    return float(np.max(np.linalg.eigvalsh((A + A.T) / 2.0)))


def geometric_difference(K_Q, K_C, lam_grid=LAM_GRID_G, g_tra_max=G_TRA_MAX):
    """Regularize geometrik fark `g` + secilen `lambda`.  D1.1.

        g(lam)     = sqrt(|| sqrt(K_Q) sqrt(K_C) (K_C+lam I)^-2 sqrt(K_C) sqrt(K_Q) ||_inf)
        g_tra(lam) = lam * sqrt(|| sqrt(K_Q) (K_C+lam I)^-2 sqrt(K_Q) ||_inf)

    `lambda` ETIKETE BAKMADAN secilir: `g_tra <= 0.045` saglayan EN BUYUK `lam`
    (Huang eq. F20). Hicbiri saglamazsa `g` TANIMSIZ raporlanir -- doldurulmaz,
    en yakin degere yuvarlanmaz (D1.1 son madde).

    Ham `K_C^-1` KULLANILMAZ: RBF Gram matrisi neredeyse tekildir, ham ters
    `g`'yi yapay sisirir.

    Doner: `(g, lam)` ya da `(nan, nan)`.
    """
    Q = trace_normalize(K_Q)
    C = trace_normalize(K_C)
    sQ, sC = _psd_sqrt(Q), _psd_sqrt(C)
    ev, V = np.linalg.eigh((C + C.T) / 2.0)
    ev = np.clip(ev, 0.0, None)
    best = (np.nan, np.nan)
    for lam in sorted(lam_grid):                       # artan -> en buyugu kalir
        R = (V * (1.0 / (ev + lam) ** 2)) @ V.T        # (K_C + lam I)^-2
        if lam * np.sqrt(max(_lmax(sQ @ R @ sQ), 0.0)) <= g_tra_max:
            best = (float(np.sqrt(max(_lmax(sQ @ sC @ R @ sC @ sQ), 0.0))), float(lam))
    return best


def spectrum_health(K):
    """`(ozdeger_min, kosul_sayisi)` -- iz-normalize `K` icin.

    `g` ile BIRLIKTE raporlanmasi sart (D2): kucuk `lam`'da regularize `g` ham
    `g`'ye yaklasir, yani sayinin ne kadar guvenilir oldugu `K_C`'nin kosul
    sayisina bagli. Tek basina `g` bunu gizler.
    """
    ev = np.clip(np.linalg.eigvalsh(trace_normalize(K)), 0.0, None)
    lo, hi = float(ev.min()), float(ev.max())
    return lo, (hi/lo if lo > 0 else float('inf'))


def model_complexity(K, y, lam):
    """`s_K = y^T (K+lam I)^-1 K (K+lam I)^-1 y` -- egitilmis modelin ||w||^2.  D1.1.

    Huang eq. F1. `lam` = o kolun KENDI LOO-secili degeri (`phaseb_core.krr_multi`),
    `g`'nin `lam`'i DEGIL: biri model karmasikligini, digeri geometrik farki
    olcuyor ve ayri secim kurallari var.

    Test hatasi kabaca `sqrt(s_K/N)` ile sinirli; `s_K ~ N` ise model hicbir sey
    ogrenmemis demektir (EC'nin imzasi).
    """
    A = trace_normalize(K)
    yv = np.asarray(to_np(y), dtype=float).ravel()
    B = np.linalg.solve(A + lam * np.eye(A.shape[0]), yv)
    return float(B @ A @ B)


# ============================================================================
# SELF-CHECK -- `python src/t11_core.py`
# ============================================================================

def _selftest():
    rng = np.random.default_rng(0)

    # [1] K1: sans EGITIM cogunlugu, test dagilimindan BAGIMSIZ
    y = np.array([1.0]*7 + [-1.0]*3)
    assert abs(chance_level(y) - 70.0) < 1e-9
    assert abs(chance_level(-y) - 70.0) < 1e-9          # simetrik
    print('[1] OK  chance_level == egitim cogunluk orani')

    # [2] K2/K3: tabakali bolme P1'i saglar; rastgele bolme SAGLAMAZ
    #     (gerekce sayisal: bu testin kendisi K2'nin varlik sebebini gosterir)
    yb = np.array([1.0]*225 + [-1.0]*126)               # ionosphere dengesi
    sp = stratified_splits(yb, 30, 20260916)
    d = [abs(chance_level(yb[tr]) - chance_level(yb[te])) for tr, te in sp]
    assert max(d) < 1.0, max(d)
    assert len(set(map(tuple, (tr for tr, _ in sp)))) == 30, 'bolmeler ayni!'
    for tr, te in sp:
        assert len(np.intersect1d(tr, te)) == 0         # P2: sizinti yok
    rnd = []
    for i in range(30):
        p = np.random.default_rng(i).permutation(len(yb)); k = int(0.7*len(yb))
        rnd.append(abs(chance_level(yb[p[:k]]) - chance_level(yb[p[k:]])))
    print(f'[2] OK  tabakali sapma max {max(d):.3f} puan  '
          f'(rastgele: {max(rnd):.2f}, std {np.std(rnd):.2f})')

    # [3] K9: kantil cikisi [0,2pi]'de ve egitim marjinali UNIFORM (P3),
    #     test kumesi fit'e GIRMIYOR (P2)
    Xr = rng.lognormal(size=(200, 12)) @ rng.normal(size=(12, 12))
    Xr = np.column_stack([Xr, np.zeros(len(Xr))])       # P4: sabit oznitelik
    A, B = preprocess(Xr[:140], Xr[140:], 8)
    assert A.shape == (140, 8) and B.shape == (60, 8)
    assert np.isfinite(A).all() and np.isfinite(B).all(), 'sabit oznitelik NaN uretti'
    assert A.min() >= 0.0 and A.max() <= 2*np.pi + 1e-9
    ks = max(float(np.max(np.abs(np.sort(A[:, j])/(2*np.pi)
                                 - (np.arange(len(A))+0.5)/len(A))))
             for j in range(A.shape[1]))
    assert ks < 0.05, ks
    A2, _ = preprocess(Xr[:140], Xr[140:][::-1], 8)     # test sirasi degisti
    assert np.allclose(A, A2), 'egitim donusumu test kumesine BAGLI -> sizinti'
    print(f'[3] OK  kantil [0,2pi], egitim KS={ks:.4f}, test fit\'e girmiyor')

    # [4] K13 kapali formu: effrank_frac == 1/(1+(M-1) m2), OLCULEN effrank ile
    for M in (245, 398, 400):
        P = rng.normal(size=(M, 3*M))
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        K = P @ P.T                                     # diag = 1
        off = K[~np.eye(M, dtype=bool)]
        pred = 1.0/(1.0 + (M-1)*float(np.mean(off**2)))
        meas = effrank(xp.linalg.eigvalsh(to_xp(K, FDT)))/M
        assert abs(pred - meas) < 1e-3, (M, pred, meas)
    print(f'[4] OK  K13 kapali formu olculen effrank ile ortusuyor (3 M degeri)')

    # [5] K13 capasi: M=400'de esik AYNEN 0.90 -> sentetik kararlar degismez;
    #     kucuk M'de SIKILASIYOR (gevsemiyor) -- yanliligin yonu kritik
    assert abs(effrank_gate_for(400) - 0.90) < 1e-12
    assert abs(effrank_gate_for(398) - 0.9005) < 1e-4
    assert abs(effrank_gate_for(245) - 0.9364) < 1e-4
    assert effrank_gate_for(245) > effrank_gate_for(400), 'kucuk M\'de kapi GEVSEDI!'
    print(f'[5] OK  esik M=400:0.900  M=398:{effrank_gate_for(398):.4f}  '
          f'M=245:{effrank_gate_for(245):.4f}')

    # [6] `g`: K_Q == K_C ise g -> 1 (ayni geometri, avantaj yok)
    #     KRITIK: `Z` DUSUK boyutlu ve bant genisligi genis secilir. Yuksek
    #     boyutta RBF'in kosegen disi terimleri e^{-d} ile sonuyor, `Kc ~ I`
    #     oluyor ve [7]/[10] "EC limiti vs klasik" ayrimini test EDEMEZ hale
    #     geliyordu (ilk surumde oyleydi: g 1.0 vs 0.976, s_K 59.9 vs 60.0).
    Z = rng.normal(size=(60, 2))
    Kc = np.exp(-0.25*((Z[:, None, :]-Z[None, :, :])**2).sum(-1)) + 1e-6*np.eye(60)
    #     Ridge KEYFI DEGIL: ham fixture'in kosul sayisi 6.3e+17 (sayisal olarak
    #     tekil), gercek veride olculen ise 3.2e+08. Ham haliyle fixture D2'nin
    #     uzatilmis izgarasinin de disinda kalip her zaman NaN veriyordu -- yani
    #     [7]'yi gercek veriyi TEMSIL ETMEYEN bir rejimde test ediyordu.
    _ev = np.linalg.eigvalsh(Kc)
    _cond = _ev.max()/_ev.min()
    assert 1e6 < _cond < 1e10, f'fixture kosul sayisi gercekci degil: {_cond:.1e}'
    g_same, lam = geometric_difference(Kc, Kc)
    assert abs(g_same - 1.0) < 0.05, (g_same, lam)
    print(f'[6] OK  K_Q==K_C -> g={g_same:.3f} (~1), lam={lam:g}')

    # [7] TUZAK (bilgi notu): EC limitinde K_Q ~ I. Iki alt hal VAR ve ikisi de
    #     `g`'nin TEK BASINA okunamayacagini gosterir -- s_Q ~ N, yani kuantum
    #     model hicbir sey ogrenmiyor, ama `g` bunu SOYLEMIYOR:
    #       (a) K_C tekil  -> hicbir lam g_tra<=0.045 saglamaz -> g TANIMSIZ
    #                         (D1.1'in "doldurulmaz" maddesi tam bu hal)
    #       (b) K_C iyi kosullu -> g BUYUK cikar ("avantaj mumkun" der, yanlis)
    yv = np.sign(rng.normal(size=60))
    sQ = model_complexity(np.eye(60), yv, 1e-3)
    assert abs(sQ - 60.0)/60.0 < 0.02, sQ                # s_Q ~ N
    #     D2'nin CALISAN GEREKCESI: ESKI izgarada (en kucuk lam=1e-5) `g`
    #     tanimsizdi; uzatilmis izgarada tanimli ve BUYUK cikiyor. Iki satir
    #     birlikte "degisen sey veri degil izgaraydi" iddiasini test eder.
    g_old, _ = geometric_difference(np.eye(60), Kc, lam_grid=(1e-5, 1e-4, 1e-3))
    assert np.isnan(g_old), g_old
    g_a, lam_a = geometric_difference(np.eye(60), Kc)
    assert np.isfinite(g_a) and g_a > 2.0*g_same, (g_a, g_same)
    g_b, _ = geometric_difference(np.eye(60), Kc + 0.05*np.eye(60))
    assert g_b > 2.0*g_same, (g_b, g_same)
    lo_, cond_ = spectrum_health(Kc)
    assert lam_a <= G_TRA_MAX*lo_*1.0001, (lam_a, lo_)   # Huang sartinin tavani
    print(f'[7] OK  EC limiti: eski izgarada TANIMSIZ, uzatilmista g={g_a:.1f} '
          f'(lam={lam_a:g} <= 0.045*{lo_:.1e}), iyi kosulluda {g_b:.1f} '
          f'-- ama s_Q={sQ:.1f} ~ N=60 (kosul {cond_:.1e})')

    # [8] `lambda` kurali: g_tra <= 0.045 sagLAYAN EN BUYUK lam secilir
    _, lam2 = geometric_difference(Kc, Kc, lam_grid=(1e-5, 1e-3))
    assert lam2 == 1e-3, lam2
    assert np.isnan(geometric_difference(Kc, Kc, lam_grid=(1e9,))[0]), 'tanimsiz olmali'
    print('[8] OK  lam = g_tra<=0.045 saglayan en buyuk; saglanmazsa NaN')

    # [9] trace normalizasyonu: kosegeni 1 olmayan kol icin Tr == N
    Kn = trace_normalize(3.7*Kc)
    assert abs(np.trace(Kn) - 60.0) < 1e-9
    assert np.allclose(Kn, trace_normalize(Kc)), 'olcek `g`\'ye siziyor'
    print('[9] OK  Tr(K)=N normalizasyonu olcegi temizliyor')

    # [10] s_K: lam->0'da K=I icin y^T y = N; guclu kernel'de KUCUK olmali
    #      "Kolay etiket" = kernel'in YUKSEK kapasiteli yonune HIZALI etiket
    #      (ust ozvektor). `sign(Z[:,0])` kullanmak yaniltici olurdu: genis
    #      bantli RBF icin o etiket puruzlu, s_K 719'a ciktigi olculdu -- yani
    #      "klasik kernel -> kucuk s_K" GENEL BIR DOGRU DEGIL, hizalanmaya bagli.
    s_id = model_complexity(np.eye(60), yv, 1e-6)
    v1 = np.linalg.eigh(trace_normalize(Kc))[1][:, -1]
    s_rbf = model_complexity(Kc, np.sqrt(60)*v1, 1e-3)
    s_hard = model_complexity(Kc, np.sign(Z[:, 0]), 1e-3)
    assert abs(s_id - 60.0) < 1e-3, s_id
    assert s_rbf < 0.1*s_id, (s_rbf, s_id)
    assert s_hard > s_id, (s_hard, s_id)
    print(f'[10] OK  s_K: K=I -> {s_id:.1f} (=N);  RBF hizali etiket -> {s_rbf:.2f};  '
          f'RBF puruzlu etiket -> {s_hard:.0f}')

    # [11] RBF/LOO secimi: ogrenilebilir etikette sansin belirgin ustunde ve
    #      secilen `gamma` medyan izgarasinin ICINDE (kenara yapismiyor)
    Xl = rng.normal(size=(160, 4))
    yl = np.sign(Xl[:, 0] + 0.5*Xl[:, 1])
    gsel, lsel, loo = rbf_select_loo(Xl, yl)
    grid = rbf_gamma_grid(Xl)
    assert loo > 0.80, loo
    assert min(grid) <= gsel <= max(grid)
    #      ogrenilemez etikette LOO sansa DUSMELI (secim etiketi ezberlemiyor)
    _, _, loo_r = rbf_select_loo(Xl, np.sign(rng.normal(size=160)))
    assert loo_r < 0.65, loo_r
    print(f'[11] OK  RBF/LOO: ogrenilebilir {loo:.2f} · rastgele etiket {loo_r:.2f} '
          f'(gamma={gsel:.3g}, lam={lsel:g})')

    # [12] `loo_lambda` == `krr_multi`'nin ICINDEKI secim.  Iki yol ayrisirsa
    #      `s_K` boru hattindakinden BASKA bir modelin karmasikligini olcer.
    #      Dogrulama: krr_multi'yi TEK lam ile cagirmak, o lam'i zorlamaktir;
    #      loo_lambda'nin sectigi lam, tek-lam cagrilarinin LOO'da en iyisi olmali.
    Kt = rbf_K(Xl[:120], Xl[:120], gsel)
    lam_b, _ = loo_lambda(Kt, yl[:120])
    lam_all = {lam: loo_lambda(Kt, yl[:120], lams=(lam,))[1] for lam in LAMS}
    assert lam_all[lam_b] == max(lam_all.values()), (lam_b, lam_all)
    print(f'[12] OK  loo_lambda tek-lam taramasinin en iyisini seciyor (lam={lam_b:g})')

    print('\nT1.1 cekirdegi: 12/12 GECTI')


if __name__ == '__main__':
    import os
    import numpy as np                                    # noqa: F811
    _h = os.path.dirname(os.path.abspath(__file__))
    G = dict(__name__='t11_core', np=np, xp=np, GPU=False,
             CDT=np.complex128, FDT=np.float64,
             to_xp=lambda a, dtype=None: np.asarray(a, dtype=dtype),
             to_np=np.asarray)
    for _f in ('qk_core.py', 'phaseb_core.py', 'gate4_core.py', 't11_core.py'):
        exec(compile(open(os.path.join(_h, _f), encoding='utf-8').read(), _f, 'exec'), G)
    G['_selftest']()
