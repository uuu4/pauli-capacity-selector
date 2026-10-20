"""Pool subsampling in the synthetic experiments (Section 5.1) -> pool_fraction.json.

The pool of terms of weight <= 4 is built once per n with qk_core.pool_layout(n, 4,
max_flips=max_flips_for(n), seed=SEED0, keep_low=2): every pattern of X/Y positions
(flip mask) with at most two X/Y positions is kept, and of the patterns with more,
max_flips are drawn at random. The pool for w_max = H is the subset of weight <= H.
This script rebuilds the pools with each experiment's own max_flips_for and SEED0
(notebooks/v3/phase-*/..._FULL.ipynb, results.json "params") and reports, per n and w_max,
the kept fraction of the non-constant terms, overall and per weight.

Usage: python make_pool_fraction.py /path/to/hope-research
"""
import json, os, sys
from math import comb
import numpy as np

root = sys.argv[1]
sys.path.insert(0, os.path.join(root, "src"))
import qk_core
qk_core.np = np

EXPS = {  # experiment: (SEED0, max_flips_for, n values)
    "1": (0, {8: 200, 10: 120, 12: 60, 14: 30}, (8, 10, 12, 14)),
    "2": (12345, {8: 200, 10: 120, 12: 60, 14: 30}, (8, 10, 12, 14)),
    "3": (777, {8: 200, 10: 120, 12: 120, 14: 60}, (8, 10, 12, 14)),
    "4": (31337, {8: 200, 10: 120, 12: 120, 14: 60}, (8, 10, 12)),
}
full_w = lambda n, w: comb(n, w) * 3 ** w
out = {}
for e, (seed, mf, ns) in EXPS.items():
    out[e] = {}
    for n in ns:
        _, _, idx, w = qk_core.pool_layout(n, 4, max_flips=mf[n], seed=seed, keep_low=2)
        w = np.asarray(w)
        assert len(np.unique(idx)) == len(idx)
        per_w = {int(k): float((w == k).sum() / full_w(n, k)) for k in range(1, 5)}
        assert per_w[1] == per_w[2] == 1.0
        out[e][n] = {f"wmax{H}": float(((w >= 1) & (w <= H)).sum() / sum(full_w(n, k) for k in range(1, H + 1)))
                     for H in (2, 3, 4)}
        out[e][n]["per_weight"] = per_w
        out[e][n]["pool_size_wmax4"] = int(((w >= 1) & (w <= 4)).sum())
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pool_fraction.json"), "w"), indent=1)
for e, d in out.items():
    for n, v in d.items():
        print(e, n, {k: round(x, 3) for k, x in v.items() if k.startswith("wmax")},
              {k: round(x, 3) for k, x in v["per_weight"].items()}, v["pool_size_wmax4"])
