"""Robustness of the Table 1 slope intervals to the bootstrap scheme -> slope_ci.json.

The fit has only nine values of n (4, 6, ..., 20), so resampling the n values with
replacement (qk_core.layer_b, used for Table 1 and for the pre-pass gates) often draws
only a few distinct n. This script recomputes every Table 1 slope with
  pairs     : the layer_b scheme (2000 resamples of the n values, redrawn if < 2 distinct n)
  residual  : residual bootstrap (2000 resamples of the least-squares residuals)
  t         : the least-squares standard error with a t interval (k - 2 degrees of freedom)
and reports, for the variance column, whether the "whole interval below zero" condition
of the exponential call agrees across the three.

Data: latifakgun/hope-research@91388cf, runs/phase_a2ext_v1/series_by_n.json (keys fid, H2, H3, H4)
(REF: ZZ map, depth 8, ent 1.0, bandwidth 0.3). Variance fits use n = 4..20, shot fits n = 8..20,
as in the main text.
Usage: python make_slope_ci.py /path/to/hope-research
"""
import json, os, sys
import numpy as np

T975 = {5: 2.571, 7: 2.365}  # t quantiles for k - 2 degrees of freedom (k = 7, 9)


def series(root, tag):
    d = json.load(open(os.path.join(root, "runs/phase_a2ext_v1/series_by_n.json")))[tag]
    return {int(n): (v["offdiag_var"], v["s_req"]) for n, v in d.items()}


def cis(ns, vals, n_boot=2000, seed=0):
    x = np.asarray(ns, float)
    y = np.log(np.asarray(vals, float))
    k = len(x)
    b, a = np.polyfit(x, y, 1)
    r = y - (a + b * x)
    rng = np.random.default_rng(seed)
    pairs = []
    while len(pairs) < n_boot:
        i = rng.integers(0, k, k)
        if len(np.unique(x[i])) >= 2:
            pairs.append(np.polyfit(x[i], y[i], 1)[0])
    rng = np.random.default_rng(seed)
    resid = [np.polyfit(x, a + b * x + r[rng.integers(0, k, k)], 1)[0] for _ in range(n_boot)]
    se = np.sqrt((r ** 2).sum() / (k - 2) / ((x - x.mean()) ** 2).sum())
    q = lambda s: [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]
    return dict(slope=float(b), se=float(se), pairs=q(pairs), residual=q(resid),
                t=[float(b - T975[k - 2] * se), float(b + T975[k - 2] * se)])


root = sys.argv[1]
out = {}
for tag, name in (("fid", "full"), ("H2", "w2"), ("H3", "w3"), ("H4", "w4")):
    s = series(root, tag)
    ns = sorted(s)
    var = cis(ns, [s[n][0] for n in ns])
    ns8 = [n for n in ns if n >= 8]
    shots = cis(ns8, [s[n][1] for n in ns8])
    var["below_zero"] = {m: var[m][1] < 0 for m in ("pairs", "residual", "t")}
    out[name] = dict(variance=var, shots=shots)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "slope_ci.json"), "w"), indent=1)
for k, v in out.items():
    for c in ("variance", "shots"):
        d = v[c]
        f = lambda iv: f"[{np.exp(iv[0]):.2f}, {np.exp(iv[1]):.2f}]"
        print(f"{k:4s} {c:8s} x{np.exp(d['slope']):.3f} pairs {f(d['pairs'])} resid {f(d['residual'])} "
              f"t {f(d['t'])}" + (f"  below0 {d['below_zero']}" if c == "variance" else ""))
