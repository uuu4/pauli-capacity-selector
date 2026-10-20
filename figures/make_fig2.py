"""Figure A1 (Appendix A): robustness to the encoding, drawn as per-qubit factors (bar charts).

(a) Full kernel under the encoding with n-independent phase spread (norm='sqrt'), depth 8, epsilon 1:
    factor by which the kernel variance is divided with every added qubit, exp(-slope), against the bandwidth c.
    Data: latifakgun/hope-research @ 7a6a3cb, runs/phase_a2r2_v1/results.json -> A0_slopes_by_bw.
(b) Per-qubit factors of the variance (divided) and of the required shots (multiplied), for the full kernel
    and the Lego kernels of weight <= 2, 3, under the original encoding (bandwidth 0.3, Section 3; variance
    slopes as in make_fig1.py, shots from sreq_slopes.json) and the n-independent encoding (bandwidth 1.0;
    variance from phase_a2r2_v1 GFQK_slopes, shots from a2r2_sreq_slopes.json). Shot fits use n >= 8.
Brackets of the slopes are 95% bootstrap confidence intervals; factors are exp(slope) for shots and
exp(-slope) for the variance.
"""
import json, math, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FULL, ORIG, NORM = "#1f4e79", "#9fb7d0", "#1f4e79"

# (a) full kernel, normalized encoding: variance slope (lo, hi) per bandwidth; exponential fit preferred?
b_bw  = [0.3, 0.6, 1.0, 1.5, 2.0, 3.0]
b_s   = [-0.364, -1.036, -1.354, -1.385, -1.392, -1.392]
b_lo  = [-0.524, -1.150, -1.377, -1.399, -1.402, -1.399]
b_hi  = [-0.206, -0.925, -1.332, -1.371, -1.383, -1.386]
b_exp = [False, True, True, True, True, True]

# (b) slopes (s, lo, hi); variance: d ln Var / dn, shots: d ln S_req / dn
VAR = {"orig": {"full": (-1.401, -1.419, -1.388), "w2": (-0.048, -0.084, -0.002), "w3": (0.039, -0.074, 0.185)},
       "norm": {"full": (-1.354, -1.377, -1.332), "w2": (-0.027, -0.071, 0.049), "w3": (0.041, -0.228, 0.299)}}
so = json.load(open(os.path.join(HERE, "sreq_slopes.json")))
sn = json.load(open(os.path.join(HERE, "a2r2_sreq_slopes.json")))
SHOT = {"orig": {k: (so[k]["slope"], so[k]["lo"], so[k]["hi"]) for k in ("full", "w2", "w3")},
        "norm": {k: (sn[k]["slope"], sn[k]["lo"], sn[k]["hi"]) for k in ("full", "w2", "w3")}}

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.8})
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw=dict(width_ratios=[1, 1.6]))

f = np.exp(-np.array(b_s)); flo = np.exp(-np.array(b_hi)); fhi = np.exp(-np.array(b_lo))
x = np.arange(len(b_bw))
for i in range(len(b_bw)):
    ax.bar(x[i], f[i], 0.6, color=FULL if b_exp[i] else "white", edgecolor=FULL, hatch=None if b_exp[i] else "///")
ax.errorbar(x, f, yerr=[f - flo, fhi - f], fmt="none", ecolor="0.25", capsize=2.5, lw=0.9)
ax.axhline(4, ls="--", color="0.35", lw=0.9); ax.text(-0.4, 4.08, "Haar: $\\div 4$", fontsize=7.5, color="0.3", va="bottom")
ax.axhline(1, color="0.6", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([str(b) for b in b_bw])
ax.set_xlabel("bandwidth $c$"); ax.set_ylabel("variance divided by, per qubit")
ax.set_ylim(0, 4.7)
ax.set_title("(a) full kernel, $n$-independent encoding", fontsize=9, loc="left")

groups = [("variance", "full", "full\nkernel"), ("variance", "w2", "weight\n$\\leq 2$"), ("variance", "w3", "weight\n$\\leq 3$"),
          ("shots", "full", "full\nkernel"), ("shots", "w2", "weight\n$\\leq 2$"), ("shots", "w3", "weight\n$\\leq 3$")]
W = 0.36
for gi, (q, k, lab) in enumerate(groups):
    xg = gi + (0.6 if q == "shots" else 0)
    for j, (enc, col) in enumerate((("orig", ORIG), ("norm", NORM))):
        s, lo, hi = (VAR if q == "variance" else SHOT)[enc][k]
        if q == "variance":
            v, vl, vh = math.exp(-s), math.exp(-hi), math.exp(-lo)
        else:
            v, vl, vh = math.exp(s), math.exp(lo), math.exp(hi)
        ax2.bar(xg + (j - 0.5) * W, v, W, color=col, edgecolor=FULL, lw=0.6,
                label=(("original encoding, $c = 0.3$" if enc == "orig" else "$n$-independent encoding, $c = 1$")
                       if gi == 0 else None))
        ax2.errorbar(xg + (j - 0.5) * W, v, yerr=[[v - vl], [vh - v]], fmt="none", ecolor="0.25", capsize=2, lw=0.8)
ax2.axhline(1, color="0.6", lw=0.8)
ax2.plot([-0.5, 2.5], [4, 4], ls="--", color="0.35", lw=0.9); ax2.plot([2.1, 5.1 + 0.0], [2, 2], ls="--", color="0.35", lw=0.9)
ax2.text(2.45, 4.08, "Haar $\\div 4$", fontsize=7, color="0.3", ha="right", va="bottom")
ax2.text(5.05, 2.08, "Haar $\\times 2$", fontsize=7, color="0.3", ha="right", va="bottom")
ax2.axvline(2.8, color="0.85", lw=0.8)
ax2.set_xticks([g + (0.6 if q == "shots" else 0) for g, (q, _, _) in enumerate(groups)])
ax2.set_xticklabels([lab for _, _, lab in groups], fontsize=7.5)
ax2.text(1.0, -1.35, "variance divided by, per qubit", ha="center", fontsize=8, transform=ax2.transData)
ax2.text(4.6, -1.35, "shots multiplied by, per qubit", ha="center", fontsize=8, transform=ax2.transData)
ax2.set_ylim(0, 4.7)
ax2.set_title("(b) both encodings", fontsize=9, loc="left")
ax2.legend(fontsize=7, frameon=False, loc="upper right")
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(os.path.join(HERE, "fig2_bandwidth.pdf")); fig.savefig(os.path.join(HERE, "fig2_bandwidth.png"), dpi=180)
print("ok")
