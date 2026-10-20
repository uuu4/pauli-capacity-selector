"""Tablo 3 (kapi orani) -- greedy-residual'in hedefe ulasmak icin tam kernel ve H-body'ye gore kac kat
daha az iki-qubit kapisina ihtiyac duydugu.

Girdi: hope-research@258df92 HOPE_runs_FULL/g1rerunv3_FULL_results.json (gercek veri izgarasinda sentetik kosu).
Satirlar: canli, EC-LEARN (yogunlasmis ama cozulebilir), qlocal etiketi HARIC (makale tum ozetlerden cikariyor).
Kapi: tam kernel cift basina bir overlap devresi (2g), secilen kernel girdi basina bir kodlama devresi (g); g sadelesir.
Ciftler: 400 egitim noktasi kendi aralarinda + 800 test x 400 egitim. Secim olcumu: 1000 etiketsiz girdi.
Calistir: python make_gate_table.py /path/to/g1rerunv3_FULL_results.json
"""
import json, math, os, sys, numpy as np
d = json.load(open(sys.argv[1]))
NTR, NTE, NSEL = 400, 800, 1000
pairs = NTR * (NTR - 1) / 2 + NTE * NTR; pts = NTR + NTE
L = [r for r in d['rows'] if r['live'] and r['ec'] == 'EC-LEARN' and r['label'] != 'qlocal']
out = dict(rows=len(L), full_never=sum(not math.isfinite(r['S_A0']) for r in L),
           both_reach=sum(math.isfinite(r['S_A0']) and math.isfinite(r['S_A7_resid']) for r in L), table={})
for lab, sel in (('exact', lambda s: 0), ('own N*', lambda s: s * NSEL), ('1000 shots', lambda s: 1000 * NSEL)):
    hb, a0 = [], []
    for r in L:
        s7, sh, s0 = r['S_A7_resid'], r['S_A2_hbody'], r['S_A0']
        den = s7 * pts + sel(s7) if math.isfinite(s7) else math.inf
        hb.append(0.0 if not math.isfinite(den) else (sh * pts) / den if math.isfinite(sh) else math.inf)
        if math.isfinite(den) and math.isfinite(s0): a0.append((s0 * pairs * 2) / den)
    out['table'][lab] = dict(vs_full=float(np.median(a0)), vs_hbody=float(np.median(hb)), n_full=len(a0))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'gate_table.json'), 'w'), indent=1)
print(json.dumps(out, indent=1))
