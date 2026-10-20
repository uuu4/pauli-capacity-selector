import json, numpy as np, collections, sys
P={'B':'phase_bv3_full_v1/phase_bv3_FULL_results.json','C':'phase_cv3_full_v1/phase_cv3_FULL_results.json',
   'C2':'phase_c2v3_full_v1/phase_c2v3_FULL_results.json','D':'phase_dv3_full_v1/phase_dv3_FULL_results.json'}
rng=np.random.default_rng(0)
def boot(rows, f, B=4000):
    cells=sorted({r['_u'] for r in rows}); by=collections.defaultdict(list)
    for r in rows: by[r['_u']].append(f(r))
    out=[]
    for _ in range(B):
        s=rng.choice(len(cells),len(cells)); v=np.concatenate([by[cells[i]] for i in s]); out.append(np.median(v))
    return np.percentile(out,[2.5,97.5])
for k,f in P.items():
    d=json.load(open(f)); rows=d['rows']
    for r in rows: r['_u']=(r['n'],r['depth'],r['ent'],r.get('bw',0.3),r.get('rep',1),r.get('jitter',0.0))
    cells={(r['n'],r['depth'],r['ent'],r.get('bw',0.3)) for r in rows}
    print(f"=== {k}: satir {len(rows)}, hucre(n,d,e,bw) {len(cells)}, bw {sorted({r.get('bw',0.3) for r in rows})}, etiket {sorted({r['label'] for r in rows})}")
    UNS=('A1_random','A2_hbody','A3_topp','A4_merge','A6_wnorm','A7_resid','A8_cost')
    for r in rows:
        if 'gold_strict' not in r:
            ch=50.0; lego=max(r[a] for a in UNS if a in r and r[a]==r[a])
            r['gold_strict']= (r['A0']<ch+3) and (lego>=ch+5)
    g=[r for r in rows if r.get('gold_strict') and r['label']!='qlocal']
    print(f"  birincil (kati altin, qlocal haric): {len(g)} satir, {len({r['_u'] for r in g})} birim, etiketler {dict(collections.Counter(r['label'] for r in g))}")
    for a,b in [('A3_topp','A2_hbody'),('A7_resid','A2_hbody'),('A3_topp','A1_random'),('A7_resid','A1_random'),('A7_resid','A3_topp'),('A3_topp','A0'),('A7_resid','A0')]:
        dif=lambda r:r[a]-r[b]; v=np.array([dif(r) for r in g])
        lo,hi=boot(g,dif)
        print(f"  {a}-{b}: medyan {np.median(v):+.2f} [{lo:+.2f},{hi:+.2f}]  pozitif {np.sum(v>0)}/{len(v)}")
    print('  mutlak medyan dogruluk:', {a: round(float(np.median([r[a] for r in g])),1) for a in ['A0','A1_random','A2_hbody','A3_topp','A7_resid','A5_super']})
