"""Ratio-criterion series for the encoding with n-independent phase spread -> a2r2_sreq_slopes.json.

Data: runs/phase_a2_v1/fig1_series_ref.csv, rows phase == phase_a2r2
(norm 'sqrt', bandwidth 1.0, depth 8, ent 1.0). Fit: qk_core.layer_b on n >= 8, the same range
convention as the main-text shot fits (sreq_slopes.json). With n >= 4 the slopes reproduce
fig1_slopes_ref.json exactly (full 0.6759, w<=2 0.0427, w<=3 0.0210).
Usage: python make_a2r2_sreq.py /path/to/hope-research
"""
import csv, json, os, sys
import numpy as np

root = sys.argv[1]
sys.path.insert(0, os.path.join(root, "src"))
import qk_core
qk_core.np = np
rows = [x for x in csv.DictReader(open(os.path.join(root, "runs/phase_a2_v1/fig1_series_ref.csv")))
        if x["phase"] == "phase_a2r2"]
key = {"A0_fidelity": "full", "GFQK_w<=2": "w2", "GFQK_w<=3": "w3"}
ser = {}
for x in rows:
    ser.setdefault(key[x["family"]], {})[int(x["n"])] = float(x["s_req"])
out = {}
for k, s in ser.items():
    ns = [n for n in sorted(s) if n >= 8]
    lb = qk_core.layer_b(ns, [s[n] for n in ns])
    out[k] = dict(ns=ns, vals=[s[n] for n in ns], slope=lb["slope"], lo=lb["slope_lo"], hi=lb["slope_hi"],
                  r2_exp=lb["r2_exp"], r2_poly=lb["r2_poly"], d_aic=lb["d_aic"])
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "a2r2_sreq_slopes.json"), "w"), indent=1)
print({k: round(v["slope"], 3) for k, v in out.items()})
