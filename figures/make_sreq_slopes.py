"""Slopes of ln S_req vs n for Figure 1(b) and Table 1 -> sreq_slopes.json.

Data: runs/phase_a2ext_v1/phase_a2_results.json
(REF: ZZ map, depth 8, ent 1.0, bandwidth 0.3; n = 8..20 for all four series).
Full kernel: regions[arm == A0_fidelity].s_req; truncated: gold[H].g_s_req.
Fit: qk_core.layer_b (2000 bootstrap resamples of the n values, seed 0, AIC vs ln n).
Usage: python make_sreq_slopes.py /path/to/hope-research
"""
import json, os, sys
import numpy as np

root = sys.argv[1]
sys.path.insert(0, os.path.join(root, "src"))
import qk_core
qk_core.np = np  # qk_core imports numpy lazily
d = json.load(open(os.path.join(root, "runs/phase_a2ext_v1/phase_a2_results.json")))
series = {"full": {r["n"]: r["s_req"] for r in d["regions"] if r["arm"] == "A0_fidelity"}}
for H in (2, 3, 4):
    series[f"w{H}"] = {r["n"]: r["g_s_req"] for r in d["gold"] if r["H"] == H}
out = {}
for k, s in series.items():
    ns = sorted(s)
    vals = [s[n] for n in ns]
    lb = qk_core.layer_b(ns, vals)
    out[k] = dict(ns=ns, vals=vals, slope=lb["slope"], lo=lb["slope_lo"], hi=lb["slope_hi"],
                  r2_exp=lb["r2_exp"], r2_poly=lb["r2_poly"], d_aic=lb["d_aic"])
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sreq_slopes.json"), "w"), indent=1)
print({k: round(v["slope"], 3) for k, v in out.items()})
