import json, numpy as np, collections
P={'B':'phase_bv3_full_v1/phase_bv3_FULL_results.json','C':'phase_cv3_full_v1/phase_cv3_FULL_results.json',
   'C2':'phase_c2v3_full_v1/phase_c2v3_FULL_results.json','D':'phase_dv3_full_v1/phase_dv3_FULL_results.json'}
rng=np.random.default_rng(0)
def boot(rows,f,B=4000):
    by=collections.defaultdict(list)
    for r in rows: by[r['_u']].append(f(r))
    cells=list(by); out=[]
    for _ in range(B):
        s=rng.choice(len(cells),len(cells)); out.append(np.median(np.concatenate([by[cells[i]] for i in s])))
    return np.percentile(out,[2.5,97.5])
for k,f in P.items():
    rows=json.load(open(f))['rows']
    for r in rows: r['_u']=(r['n'],r['depth'],r['ent'],r.get('bw',0.3),r.get('rep',1),r.get('jitter',0.0))
    base=[r for r in rows if r['label'] not in ('qlocal',)]
    sets={'A0 olu (tumu)':[r for r in base if r['A0']<53],
          'A0 olu & oracle canli':[r for r in base if r['A0']<53 and r['A5_super']>=55],
          'tum canli':[r for r in base if r['live']]}
    for nm,g in sets.items():
        if not g: print(k,nm,'0'); continue
        out=[]
        for a,b in [('A3_topp','A2_hbody'),('A7_resid','A2_hbody'),('A3_topp','A1_random')]:
            f_=lambda r:r[a]-r[b]; v=np.array([f_(r) for r in g]); lo,hi=boot(g,f_)
            out.append(f"{a[:2]}-{b[:2]} {np.median(v):+.2f} [{lo:+.2f},{hi:+.2f}] {np.sum(v>0)}/{len(v)}")
        lab=dict(collections.Counter(r['label'] for r in g))
        print(f"{k:3s} {nm:24s} n={len(g):4d} | "+' | '.join(out)+f" | {lab}")
