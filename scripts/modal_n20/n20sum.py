import json, numpy as np, glob
def load(f): return json.load(open(f))
R={'io20':'n20res/t11iobign20v3_FULL_results.json','fm20':'n20res/t11fmbign20v3_FULL_results.json'}
for k,f in R.items():
    d=load(f); print('=====',k,'classical:',{a:(round(b,1) if isinstance(b,(int,float)) else b) for a,b in list(d['classical'].items())[:6]} if isinstance(d['classical'],dict) else d['classical'])
    print(' ec:', d['ec'] if not isinstance(d['ec'],dict) else {u:v for u,v in list(d['ec'].items())[:6]})
    for r in d['rows']:
        f2=lambda a: r.get('acc_'+a)
        S=lambda a: r.get('S_'+a)
        print(f" n{r['n']} e{r.get('ent',r.get('eps'))} bw{r['bw']} H{r['H']} p{r['p']} ch{r['chance']:.1f} live={r.get('live')} ec={r.get('ec')} k6={r.get('k6_ok')} | A0 {f2('A0'):.1f} Hb {f2('A2_hbody'):.1f} top {f2('A3_topp'):.1f} gr {f2('A7_resid'):.1f} orc {f2('A5_super'):.1f} rnd {f2('A1_random'):.1f} | S* A0 {S('A0')} Hb {S('A2_hbody')} gr {S('A7_resid')}")
