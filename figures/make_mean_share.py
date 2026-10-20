"""Share of the mean in the capacity of the non-constant Pauli terms (Section 2.4) -> mean_share.json.

sum_i (E_x psi_i)^2 = tr(rho_bar^2) = E_{x != x'} k(x, x'), the mean off-diagonal value of the
full fidelity kernel, and sum_i gamma_i = 1 with gamma_I = 2^-n. Over the non-constant terms the
share of the mean is therefore (mean off-diagonal k - 2^-n) / (1 - 2^-n). Checked against the
term-by-term sum (check_capacity_mean.py) at n = 4 and 6 to machine precision.

Data: runs/phase_a2ext_v1/series_by_n.json (key fid)
(REF: ZZ map, depth 8, ent 1.0, bandwidth 0.3; M = 400 / 250 / 200 points).
Usage: python make_mean_share.py /path/to/hope-research
"""
import json, os, sys

root = sys.argv[1]
out = {}
for n, v in json.load(open(os.path.join(root, "runs/phase_a2ext_v1/series_by_n.json")))["fid"].items():
    n = int(n)
    out[n] = dict(M=v["M"], offdiag_mean=v["offdiag_mean"],
                  mean_share=(v["offdiag_mean"] - 2.0 ** -n) / (1 - 2.0 ** -n))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mean_share.json"), "w"), indent=1)
print({n: f"{100 * d['mean_share']:.3g}%" for n, d in out.items()})
