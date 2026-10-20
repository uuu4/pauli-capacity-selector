"""Faz B cekirdegi -- ogrenme tarafi. `qk_core.py` ile birlikte deftere gomulur.

qk_core: devre, Pauli domeni, kernel, EC teshisi (Katman A/B/C)
phaseb : etiketler, kol secimi (A1-A5), merge, KRR, istatistik, Katman D

BEKLENEN ORTAM: qk_core'un tanimladigi her sey + `xp/GPU/CDT/FDT/to_xp/to_np`.
"""


# ============================================================================
# 1) ETIKET BATARYASI -- SABIT koordinat alt kumesinde
# ============================================================================
# KRITIK: her etiket sadece ilk `n_fixed` koordinata bakar. n buyurken eklenen
# qubitler SEYIRCI olur -> gorevin icsel zorlugu SABIT kalir -> dogruluk dususu
# SADECE EC'ye atfedilebilir. (v1'de 'parity' tum n'e bakiyordu, iki etki
# karisiyordu -- 02_PROJECT_STATUS.md 5.7 madde 2.)
#
# `qglobal` BILEREK istisna: n ile zorlasir, "EC referansi" olarak raporlanir.

LABELS = ['lin3', 'quad3', 'parity3', 'trig3', 'qlocal', 'qglobal']


def make_labels(X, kind, n, seed, states=None, n_fixed=3):
    rng = np.random.default_rng(seed + 5000)
    k = min(n_fixed, X.shape[1])
    x = X[:, :k]
    if kind == 'lin3':
        vals = np.sin(x) @ rng.normal(size=k)
    elif kind == 'quad3':
        vals = np.sin(x[:, 0])*np.sin(x[:, 1]) + 0.5*np.cos(x[:, 2 % k])
    elif kind == 'parity3':
        # ZATEN +-1 degerli -> medyan-merkezleme UYGULANMAZ (self-test yakaladi:
        # medyan +-1 cikinca sign(vals-medyan) her seyi tek sinifa cevirir).
        y = np.prod(np.sign(np.sin(x)), axis=1).astype(float)
        y[y == 0] = 1.0
        return y
    elif kind == 'trig3':
        vals = np.sin(x[:, 0] + x[:, 1])*np.cos(x[:, 2 % k])
    elif kind == 'qlocal':                         # ilk 2 qubitte etkili gozlenebilir
        assert states is not None
        d = 2**n
        A = rng.normal(size=(4, 4)) + 1j*rng.normal(size=(4, 4))
        O = (A + A.conj().T)/2
        s = to_np(states).reshape(-1, 4, d // 4)
        vals = np.real(np.einsum('mir,ij,mjr->m', s.conj(), O, s))
    elif kind == 'qglobal':
        # GLOBAL gozlenebilir: O = sum_j c_j P_j, P_j = UNIFORM RASTGELE AGIRLIK-n Pauli.
        #
        # ONCEKI SURUM 2^n x 2^n YOGUN MATRIS URETIYORDU -> n=16'da 68 GB, Colab
        # oturumu cokuyordu; n=12'de de tohum basina ~3 x (M x 4096 x 4096) einsum
        # ile surenin cogunu yiyordu. Bu surum matematiksel olarak ayni sinifta
        # (tum qubitlere dokunan gozlenebilir) ama maliyeti O(R M n 2^n).
        assert states is not None
        R = 8
        # agirlik-n Pauli: her qubitte X/Y/Z (identity YOK) -> basamak in {1,2,3}
        idx = np.array([sum(int(dd)*(4**k) for k, dd in enumerate(rng.integers(1, 4, n)))
                        for _ in range(R)], dtype=np.int64)
        cj = rng.normal(size=R)
        # R cekilis CAKISABILIR (n=8'de ~%0.4). `pool_layout_from_indices` unique
        # uyguladigi icin sutun sayisi R'den az olur -> eski surumde IndexError.
        # Cakisan terimlerin katsayilarini TOPLA: sum_j c_j P_j degismez.
        uniq, inv = np.unique(idx, return_inverse=True)
        c_agg = np.zeros(len(uniq))
        np.add.at(c_agg, inv, cj)
        lay = pool_layout_from_indices(uniq, n)
        E = to_np(pauli_expect_fast(states, n, lay))   # sutunlar artan Pauli indeksi
        vals = E @ c_agg                               # == uniq sirasi, yeniden siralama YOK
    else:
        raise ValueError(kind)
    y = np.sign(vals - np.median(vals))
    y[y == 0] = 1.0
    return y


# ============================================================================
# 2) KOL SECIMI  A1-A5
# ============================================================================

def capacity(E_sel):
    """gamma_i = E_x[psi_i(x)^2]  -- ETIKETSIZ kapasite (bizim kriterimiz)."""
    return to_np(xp.mean(E_sel**2, axis=0))


def select_random(n_pool, p, rng, n_draws=5):
    """A1 -- Gan et al.'in fiilen yaptigi: uniform rastgele."""
    return [rng.choice(n_pool, size=min(p, n_pool), replace=False) for _ in range(n_draws)]


def select_hbody(w_pool, p, rng, n_draws=5):
    """A2_low -- DUSUK AGIRLIK ONCELIKLI rastgele secim: agirlik<=H operatorleri UNIFORM
    ele alir, en kucuk H'den baslar. KAPASITE BILGISI KULLANMAZ.

    ⚠️ 2026-10-15 DUZELTME: Bu, Gan+'in H-body LPQK'si DEGILDIR (2311.13552 Eq. 42-43:
    agirligi TAM H olan Pauli'ler, I yok; deneyde tam 2-body havuzdan rastgele p).
    Faz B-D ve T1.1/FM kosu 1'de "A2_hbody" adiyla bu fonksiyon kullanildi ve Gan+'in
    deney rakibinden GUCLU cikti (runs/rival_check_v1). Gan+'a sadik kol: `select_hbody_exact`.

    v1 HATASI: kapasite-sirali listeden `cand[:p]` aliyordu -> A2 gizlice kapasite
    kullaniyordu ve H=H_max'te A3 ile OZDESLESIYORDU (hata #12).
    """
    w = np.asarray(w_pool)
    for H in range(1, int(w.max()) + 1):
        cand = np.where(w <= H)[0]
        if len(cand) >= p:
            if len(cand) == p:
                return [cand]
            return [rng.choice(cand, size=p, replace=False) for _ in range(n_draws)]
    cand = np.arange(len(w))
    return [rng.choice(cand, size=min(p, len(cand)), replace=False) for _ in range(n_draws)]


def select_hbody_exact(w_pool, h, p, rng, n_draws=5):
    """A2_hb{h} -- Gan+'a SADIK H-body (2311.13552 Eq. 42-43, Sec. IV): agirligi TAM `h`
    olan operatorlerden UNIFORM rastgele p. I (agirlik 0) hicbir zaman girmez.

    Gan+ deneyde `h = 2`, "averaged over 10 different random selections of p features".
    Havuz `p`'den kucukse BOS liste doner (kol o satirda tanimsiz); esitse tek secim.
    """
    cand = np.where(np.asarray(w_pool) == h)[0]
    if len(cand) < p:
        return []
    if len(cand) == p:
        return [cand]
    return [rng.choice(cand, size=p, replace=False) for _ in range(n_draws)]


def select_topp(gamma, p):
    """A3 -- BIZIM kriterimiz: kapasiteye gore top-p (veri-bagimsiz, etiketsiz)."""
    return np.argsort(gamma)[::-1][:p]


def weight_normalized_capacity(gamma, w):
    """gamma~_i = gamma_i / E[gamma | agirlik(i)]  -- AGIRLIK YANLILIGI duzeltmesi.

    TARAMA BULGUSU (runs/screen_v1): 157 sinyalli hucrede top-16'nin agirlik profili
        BIZ (gamma):  w1:0.74  w2:0.19  w3:0.01  w4:0.00
        ORACLE (KTA): w1:0.11  w2:0.42  w3:0.31  w4:0.16
    Sinyal w2-w4'te, biz w1'e bakiyoruz. Sebep: yapili durumlarda dusuk agirlikli
    gozlenebilirlerin beklenen degerleri DOGAL OLARAK buyuk; gamma mutlak buyuklugu
    olcuyor, bilgi tasiyip tasimadigini degil. Ayrica bu, A3'un A2 (H-body, o da
    dusuk agirlik seciyor) ile ozdeslesmesini ve ona kaybetmesini acikliyor.

    Duzeltme: her operatoru KENDI AGIRLIK SINIFINDAKI tipik degere gore sirala.
    Etiketsiz kalir (sadece gamma ve agirlik kullanir) -> A5'in aksine ust sinir degil.
    """
    g = np.asarray(gamma, dtype=float)
    w = np.asarray(w)
    out = np.zeros_like(g)
    for ww in np.unique(w):
        m = (w == ww)
        mu = g[m].mean()
        out[m] = g[m]/mu if mu > 0 else 0.0
    return out


def select_topp_wnorm(gamma, w, p):
    """A6 -- agirlik-normalize kapasiteye gore top-p."""
    return np.argsort(weight_normalized_capacity(gamma, w))[::-1][:p]


# --- KAPI 3: HOPE'un gercek kurali  argmin J_k / dP_k -------------------------

def select_greedy_residual(E_sel, p, cost=None, block=4096):
    """A7/A8 -- ARTIK kapasiteye gore acgozlu secim (HOPE'un rate-distortion kurali).

    `gamma_i = E[psi_i^2]` bir operatorun MUTLAK katkisini olcer. HOPE'un kurali ise
    marjinaldir: bir birimin degeri, ZATEN TUTULANLARIN USTUNE ne kattigidir.
    Burada bu, secilenlerin gerdigi altuzaya dik bilesenin kapasitesidir:

        gamma_i^res = || psi_i - P_S psi_i ||^2 / M ,   S = simdiye kadar secilenler

    Her adimda `argmax gamma^res` secilir (= `argmin J_k/dP_k`, dP_k = 1 operator).
    `argmax gamma` bunun S=bos ozel halidir -- yani A3, bu kuralin ILK ADIMIDIR ve
    redundansi hic gormez. Tarama v2 bu korlugun bedelini olctu: A6 agirlik profilini
    duzeltti ama ortusme %6'da kaldi (bkz. 02_PROJECT_STATUS.md 6.995).

    `cost` verilirse marjinal kapasite ONA BOLUNUR (A8): tam `J_k/dP_k` formu,
    `dP_k` = operatorun olcum maliyeti (classical shadow: 3^w).
    UYARI: 3^w cok hizli buyudugu icin A8'in dusuk agirliga cokmesi BEKLENIR --
    yani A3'un hatasini tekrarlayabilir. Ikisi de olculup raporlanir.

    ETIKETSIZ kalir: yalnizca secim setindeki `psi` degerlerini kullanir.
    """
    R = xp.array(E_sel, copy=True)                    # (M, K) artiklar
    M, K = R.shape
    nrm = xp.sum(R**2, axis=0) / M
    c = None if cost is None else to_xp(np.asarray(cost, dtype=np.float64), FDT)
    chosen = []
    for _ in range(min(p, K)):
        score = nrm if c is None else nrm / xp.maximum(c, 1e-30)
        j = int(to_np(xp.argmax(score)))
        if float(to_np(nrm[j])) <= 1e-24:             # kalan her sey zaten kapsanmis
            break
        chosen.append(j)
        q = R[:, j] / xp.sqrt(xp.sum(R[:, j]**2))
        for s in range(0, K, block):                  # bellek dostu izdusum
            sl = slice(s, min(s+block, K))
            R[:, sl] -= xp.outer(q, q @ R[:, sl])
        nrm = xp.sum(R**2, axis=0) / M
        nrm[to_xp(np.asarray(chosen))] = 0.0
    # v2 (2026-10-15): rank < p ise kalan yer secilmemislerden ham kapasite sirasiyla
    # doldurulur. Eskiden `break` p'den kisa liste donduruyordu -> kollar ust uste
    # konamayip Faz B v2 dustu (n=14, depth=1, ent=0, H=2: rank 8). Doldurulanlar
    # secilenlerin gerdigi altuzayda (artik ~0), kernel'e yeni yon katmaz.
    if len(chosen) < min(p, K):
        cap = to_np(xp.sum(E_sel**2, axis=0)).astype(float)
        cap[np.asarray(chosen, dtype=int)] = -np.inf
        chosen += [int(j) for j in np.argsort(-cap, kind='stable')[:min(p, K) - len(chosen)]]
    return np.asarray(chosen)


def select_kta(E_sel, y_sel, p):
    """A5 -- DENETIMLI ust sinir (kernel-target alignment).
    KTA_i = (sum_m psi_i(x_m) y_m)^2 / sum_m psi_i(x_m)^2.  Etiket KULLANIR."""
    y = to_xp(np.asarray(y_sel, dtype=np.float64), FDT)
    num = (y[None, :] @ E_sel).ravel()**2
    den = xp.sum(E_sel**2, axis=0) + 1e-30
    return to_np(xp.argsort(num/den)[::-1][:p])


# ============================================================================
# 3) MERGE -- genellestirilmis Eckart-Young, SADECE etiketsiz sette ogrenilir
# ============================================================================

def greedy_merge(E_sel, idx, keep):
    """En korele cifti tekrar tekrar birlestirir; `keep` yon kalana dek.

    G = [[g_i, m_ij],[m_ij, g_j]] -> birlesik kapasite = lam1, distorsiyon = lam2,
    yon = ust ozvektor.  Doner: (C, distorsiyon)  --  C: (len(idx), keep) katsayi
    matrisi; ozellikler F = E[:, idx] @ C.

    KRITIK: C SADECE etiketsiz secim setinden ogrenilir, sonra tr/te'ye UYGULANIR.
    """
    E = E_sel[:, to_xp(np.asarray(idx))]
    G = to_np(E.T @ E) / E.shape[0]                  # (K,K) Gram
    K = G.shape[0]
    C = np.eye(K)
    dist = 0.0
    while C.shape[1] > keep:
        Gc = C.T @ G @ C                             # mevcut yonlerin Gram'i
        d = np.sqrt(np.maximum(np.diag(Gc), 1e-30))
        R = np.abs(Gc / np.outer(d, d))              # normalize korelasyon
        np.fill_diagonal(R, -1.0)
        i, j = np.unravel_index(np.argmax(R), R.shape)
        sub = Gc[np.ix_([i, j], [i, j])]
        ev, V = np.linalg.eigh(sub)                  # artan sirada
        dist += float(ev[0])                         # lam2 = kaybedilen kapasite
        new = C[:, i]*V[0, 1] + C[:, j]*V[1, 1]      # lam1 yonu
        C = np.delete(C, [i, j], axis=1)
        C = np.column_stack([C, new])
    return C, dist


# ============================================================================
# 4) KERNEL + KRR  (ozayrisimla lambda taramasi ~bedava)
# ============================================================================

def kernel_pair(F_a, F_b, diag_a=None, diag_b=None):
    """Birim kosegene normalize kernel. F: (M, p) ozellikler."""
    da = xp.maximum(xp.sum(F_a**2, 1), 1e-30) if diag_a is None else diag_a
    db = xp.maximum(xp.sum(F_b**2, 1), 1e-30) if diag_b is None else diag_b
    return (F_a @ F_b.T) / xp.sqrt(xp.outer(da, db))


LAMS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def krr_multi(Ktr, Kte, Ytr, Yte, lams=LAMS):
    """Kernel ridge, TUM etiketler tek cagride, `lambda` EGZAKT LOO-CV ile secilir.

    Neden boyle: kernel etiketten BAGIMSIZ, yani 6 etiket ayni ayrisimi paylasabilir.
    Ktr = U diag(ev) U^T BIR KEZ hesaplanir; sonra her lambda ve her etiket icin
    cozum sadece O(M^2) matris-vektor.

    LOO kapali formu (kat dongusu YOK, hem daha hizli hem daha iyi istatistik):
        H = U diag(ev/(ev+lam)) U^T,   h_i = H_ii
        yhat_i^{-i} = (yhat_i - h_i y_i) / (1 - h_i)

    Ilk surumde her (etiket, lambda, kat) icin ayri `solve` vardi -> ~112k cagri,
    pilot saatlerce surdu. Bu surum ayni sonucu ~2 ozayrisimla veriyor.

    Doner: {etiket_adi: (dogruluk_yuzde, correct_vec)}
    """
    Ktr = Ktr if Ktr.ndim == 3 else Ktr[None]      # (B,M,M)
    Kte = Kte if Kte.ndim == 3 else Kte[None]      # (B,Mte,M)
    names = list(Ytr.keys())
    Y = to_xp(np.column_stack([Ytr[k] for k in names]).astype(np.float64), FDT)   # (M,L)
    B, L = Ktr.shape[0], len(names)

    ev, U = xp.linalg.eigh(Ktr)                    # YIGIN halinde: tek cagri
    ev = xp.clip(ev.real, 0, None)                 # (B,M)
    UtY = xp.matmul(xp.swapaxes(U, 1, 2), Y[None])  # (B,M,L)  = U^T Y
    U2 = U**2
    best_acc = xp.full((B, L), -1.0, dtype=FDT)
    best_lam = np.zeros((B, L))
    for lam in lams:
        d = 1.0/(ev + lam)
        f = (ev*d)[:, :, None]                     # (B,M,1)
        yhat = xp.matmul(U, f*UtY)
        h = xp.matmul(U2, (ev*d)[:, :, None])
        loo = (yhat - h*Y[None]) / xp.maximum(1.0 - h, 1e-6)
        acc = xp.mean(xp.sign(loo) == Y[None], axis=1)     # (B,L)
        upd = acc > best_acc
        best_lam = np.where(to_np(upd), lam, best_lam)
        best_acc = xp.where(upd, acc, best_acc)

    lam_x = to_xp(best_lam, FDT)                   # (B,L)
    d = 1.0/(ev[:, :, None] + lam_x[:, None, :])   # (B,M,L)
    A = xp.matmul(U, d*UtY)                        # (B,M,L)  katsayilar
    P = to_np(xp.sign(xp.matmul(Kte, A)))          # (B,Mte,L)
    out = {}
    for b in range(B):
        for j, k in enumerate(names):
            c = (P[b, :, j] == np.asarray(Yte[k])).astype(np.float64)
            out[(b, k) if B > 1 else k] = (float(c.mean()*100), c)
    return out


def krr_accuracy(Ktr, Kte, ytr, yte, lams=LAMS):
    """Tek etiketlik ince sarmalayici (dogrulama testleri icin)."""
    r = krr_multi(Ktr, Kte, {'y': ytr}, {'y': yte}, lams)
    return r['y']


# ============================================================================
# 5) KATMAN D -- ogrenilebilirlik on-kosulu  (ON-KAYITLI)
# ============================================================================
# Faz A/A2/A2R2'nin tamami bu adima bagli. "w<=2 kernel konsantre olmuyor" tek
# basina UCUZ bir sonuc -- sabit bir kernel de konsantre olmaz. Bulgunun anlam
# kazanmasi icin ayni kernel'in HALA OGRENDIGI gosterilmelidir.

D_LIVE   = 5.0    # bir hucrenin karar tablosuna girmesi icin: en iyi kol >= sans+5
D_A0DEAD = 3.0    # "A0 olu" esigi: A0 < sans+3


def layer_d(acc_by_arm, chance=50.0):
    """A0 x Lego 2x2 tablosu. Doner: (durum, canli_mi, altin_mi)

    |                | Lego canli                  | Lego olu        |
    |----------------|-----------------------------|-----------------|
    | **A0 olu**     | `EC-ACTIVE-LEGO-SURVIVES` ✅ | `DEAD`          |
    | **A0 canli**   | `EC-MILD`                   | `LEGO-LOSES` ⚠  |

    `EC-ACTIVE-LEGO-SURVIVES` -> ALTIN, EC-mitigasyon iddiasi icin GECERLI tek rejim.
    `EC-MILD`     -> A0 saglikli; buradaki kazanc EC-mitigasyon KANITI DEGIL.
    `LEGO-LOSES`  -> tam kernel'de sinyal VAR, biz kaybediyoruz. **Bizim ALEYHIMIZE
                     sonuc, durustce raporlanmali.** (v1'in yaniltici 'EC-ACTIVE-ALL-DEAD'
                     etiketi bu durumu gizliyordu.)
    `DEAD`        -> hicbir kol ogrenmiyor, hucre bilgi tasimiyor -> dislanir.
    `AMBIGUOUS`   -> A0, [sans+3, sans+5) bandinda; ne olu ne canli sayilir, dislanir.
    """
    a0 = acc_by_arm.get('A0', 0.0)
    lego = max(v for k, v in acc_by_arm.items() if k != 'A0')
    a0_dead = a0 < chance + D_A0DEAD
    a0_alive = a0 >= chance + D_LIVE
    lego_alive = lego >= chance + D_LIVE
    if not (a0_alive or lego_alive):
        return 'DEAD', False, False
    if a0_dead and lego_alive:
        return 'EC-ACTIVE-LEGO-SURVIVES', True, True
    if a0_alive and lego_alive:
        return 'EC-MILD', True, False
    if a0_alive and not lego_alive:
        return 'LEGO-LOSES', True, False
    return 'AMBIGUOUS', False, False


# --- HATA #29 duzeltmesi (2026-08-17) -------------------------------------
# Faz B FULL'de altin hucrelerin %87'sinde `>=sans+5` sartini SADECE `A5_super`
# (etiket kullanan oracle) sagliyordu. Yani sart "bilgi hala orada" diyordu,
# "bizim ETIKETSIZ yontemimiz calisiyor" DEMIYORDU. KATI ALTIN bunu kapatir.
# Faz C'de ON-KAYITLI birincil kapi budur (01_TEST_PROTOCOL.md 6.37).
STRICT_UNSUP = ('A1_random', 'A2_hbody', 'A3_topp', 'A4_merge', 'A6_wnorm', 'A7_resid',
                'A8_cost',   # A8 KAPI 4'te acildi (2026-08-21); eski kosularda bu
                             # anahtar hic uretilmedigi icin Faz A-D sonuclari AYNI
                'A2_hb1', 'A2_hb2', 'A2_low')   # v2 rakipleri (2026-10-15), ayni mantik


def layer_d_strict(acc_by_arm, chance=50.0):
    """Katman D, ama "Lego hayatta" yarisini SADECE ETIKETSIZ bir kol saglayabilir.

    `layer_d` ile ayni 2x2; tek fark `A5_super` (ve baska her denetimli kol)
    hayatta-kalma oyuna KATILMAZ. `A0` her zaman kalir (karsilastirma ekseni).
    """
    sub = {k: v for k, v in acc_by_arm.items() if k == 'A0' or k in STRICT_UNSUP}
    if len(sub) < 2:
        raise ValueError('layer_d_strict: etiketsiz kol yok')
    return layer_d(sub, chance)


def redundancy(E_sel, idx):
    """Secilen `psi` sutunlari arasinda ortalama |Pearson korelasyonu| (kosegen haric).

    A7/A4'un mekanizmasi REDUNDANSA dayanir: artik-kapasite ve merge, ancak
    sutunlar korele oldugunda ham `gamma`'dan farklilasir. Bu olcum E2'nin
    (en guclu etki rejiminin taranmasi) OLCULEBILIR halidir -- tarama v3
    "A7 == A3" dedi ama redundansin O REJIMDE var olup olmadigini olcmemisti.
    """
    F = E_sel[:, to_xp(np.asarray(idx))]
    F = F - xp.mean(F, 0)
    s = xp.sqrt(xp.sum(F**2, 0))
    C = to_np((F.T @ F) / xp.maximum(xp.outer(s, s), 1e-30))
    k = C.shape[0]
    if k < 2:
        return 0.0
    return float(np.abs(C[~np.eye(k, dtype=bool)]).mean())


# ============================================================================
# 6) ISTATISTIK -- 01_TEST_PROTOCOL.md esikleri, DEGISTIRILMEDI
# ============================================================================

WIN_THR, ELIM_THR = 1.0, 0.3


def paired_bootstrap(A, B, n_boot=4000, seed=0):
    """A, B: (n_seed, n_test) dogru/yanlis matrisleri. Tohum duzeyinde BLOK
    yeniden orneklemeli eslestirilmis bootstrap."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    D = A - B
    rng = np.random.default_rng(seed)
    ns, nt = D.shape
    point = float(D.mean()*100)
    st = np.empty(n_boot)
    for t in range(n_boot):
        s = rng.integers(0, ns, ns); i = rng.integers(0, nt, nt)
        ws = np.bincount(s, minlength=ns)
        wi = np.bincount(i, minlength=nt)
        st[t] = (ws @ D @ wi) / (ns * nt) * 100.0
    return point, float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))


def verdict(point, lo, hi, win=WIN_THR, elim=ELIM_THR):
    """Iki esik ailesi (hata #3): kazanc ve eleme esikleri TERS yone calisir."""
    if lo > 0 and point >= win:
        return 'KAZANC'
    if hi < elim:
        return 'ELEME-ADAYI'
    return 'SONUCSUZ'


COMPARISONS = {
    # --- protokolun ana sarti: A3/A4 hem A1'i hem A2'yi yensin, >=2 etikette ---
    'A3 top-p vs A1 rand':  ('A3_topp',  'A1_random'),
    'A3 top-p vs A2 hbody': ('A3_topp',  'A2_hbody'),
    'A4 merge vs A1 rand':  ('A4_merge', 'A1_random'),
    'A4 merge vs A2 hbody': ('A4_merge', 'A2_hbody'),   # v1'de EKSIKTI (hata #13)
    'A4 merge vs A3 top-p': ('A4_merge', 'A3_topp'),
    # --- Kapi 3 kollari (tarama v3'ten sonra eklendi; ELEME icin E2 eksik) ---
    'A7 resid vs A2 hbody': ('A7_resid', 'A2_hbody'),
    'A7 resid vs A1 rand':  ('A7_resid', 'A1_random'),
    'A7 resid vs A3 top-p': ('A7_resid', 'A3_topp'),
    'A7 resid vs A4 merge': ('A7_resid', 'A4_merge'),   # redundansa iki farkli cevap
    'A6 wnorm vs A3 top-p': ('A6_wnorm', 'A3_topp'),
    # --- Faz C: EC-mitigasyon iddiasinin ASIL cumlesi (2026-08-17) ---
    # "A0 olu, biz canli" bir REJIM etiketiydi (Katman D); bu ise DOGRUDAN,
    # eslestirilmis, GA'li bir karsilastirma. Iddia icin en gucluk tek satir.
    'A3 top-p vs A0 fidel': ('A3_topp',  'A0'),
    'A7 resid vs A0 fidel': ('A7_resid', 'A0'),
    # --- ust sinir ---
    # HATA #32 (2026-08-19): tek yonlu karsilastirma "A3 anlamli USTUN" vakalarini
    # gorunmez kiliyordu -- elle hesaplamak zorunda kaldim. Simdi iki yonlu.
    'A3 top-p vs A7 resid': ('A3_topp',  'A7_resid'),
    'A5 super vs A3 top-p': ('A5_super', 'A3_topp'),
    'A5 super vs A7 resid': ('A5_super', 'A7_resid'),
}
