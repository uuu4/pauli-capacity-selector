"""Figure 1 (Section 3.2-3.3): the two diagnostic criteria at the reference setting.

(a) Per-qubit decay rate of the kernel variance, d ln Var[k]/dn, 95% bootstrap CI.
    Data: runs/phase_a2_v1/results.json
    -> slopes_fid ("depth|ent"), full kernel n = 4..20. Truncated (Lego) rows:
    runs/phase_a2ext_v1/phase_a2_results.json -> slopes_ser[H]["8|1.0|0.0"], n = 4..20
    (extension run, 2026-10-16; full-kernel values reproduce phase_a2_v1 to 5e-7).
    ZZ map, depth 8, bandwidth 0.3.
    Hollow markers: exponential fit not preferred (Delta AIC > 0 or CI reaches 0).
(b) Required shots per kernel entry (ratio criterion), S_req = sigma^2_shot / Var[k].
    Data: phase_a2ext_v1 -> regions (full) and gold (truncated), REF, n = 8..20;
    slopes in sreq_slopes.json, written by make_sreq_slopes.py (qk_core.layer_b).
"""
import json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
HAAR = -math.log(4)
FULL, LEGO = "#1f4e79", "#c0504d"

a_rows = [  # label, slope, lo, hi, exponential
    ("Full kernel, $\\varepsilon = 0$",    -0.656, -0.699, -0.625, True),
    ("Full kernel, $\\varepsilon = 0.25$", -1.194, -1.212, -1.184, True),
    ("Full kernel, $\\varepsilon = 0.5$",  -1.329, -1.353, -1.315, True),
    ("Full kernel, $\\varepsilon = 1$",    -1.401, -1.419, -1.388, True),
    (r"Lego, Pauli weight $\leq 2$", -0.048, -0.084, -0.002, False),
    (r"Lego, Pauli weight $\leq 3$", 0.039, -0.074, 0.185, False),
    (r"Lego, Pauli weight $\leq 4$", -0.006, -0.263, 0.262, False),
]
d = json.load(open(os.path.join(HERE, "sreq_slopes.json")))

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.8})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.15, 1]})

for i, (lab, s, lo, hi, ex) in enumerate(a_rows):
    y = len(a_rows) - 1 - i
    c = FULL if lab.startswith("Full") else LEGO
    ax1.errorbar(s, y, xerr=[[s - lo], [hi - s]], fmt="o", color=c, ms=4.5, capsize=2.5, lw=1,
                 mfc=c if ex else "white")
ax1.axvline(HAAR, ls="--", color="0.35", lw=0.9)
ax1.axvline(0, color="0.75", lw=0.8)
ax1.text(HAAR + 0.03, -0.45, r"Haar $-\ln 4$", ha="left", va="bottom", fontsize=8, color="0.3")
ax1.set_yticks(range(len(a_rows)))
ax1.set_yticklabels([r[0] for r in a_rows][::-1], fontsize=8)
ax1.set_ylim(-0.6, len(a_rows) - 0.1)
ax1.set_xlim(-1.6, 0.4)
ax1.set_xlabel(r"variance decay rate  $d\,\ln \mathrm{Var}[k]\,/\,dn$")
ax1.set_title("(a) Variance", fontsize=9)

style = {"full": (r"Full kernel, $\varepsilon = 1$", FULL, "o"),
         "w2": (r"Lego, weight $\leq 2$", LEGO, "s"),
         "w3": (r"Lego, weight $\leq 3$", "#e08b3e", "^"),
         "w4": (r"Lego, weight $\leq 4$", "#8064a2", "D")}
for k, (lab, c, m) in style.items():
    v = d[k]
    ax2.plot(v["ns"], v["vals"], marker=m, color=c, ms=4, lw=1, label=f"{lab} ({v['slope']:+.2f}/qubit)")
ax2.axhline(1e5, ls=":", color="0.4", lw=0.9)
ax2.text(8.1, 1.25e5, r"budget $10^5$", fontsize=7.5, color="0.35")
ax2.set_yscale("log")
ax2.set_ylim(5, 2e6)
ax2.set_xticks(range(8, 21, 2))
ax2.set_xlabel("number of qubits $n$")
ax2.set_ylabel("required shots per entry")
ax2.legend(fontsize=6.3, frameon=False, loc="lower right")
ax2.set_title("(b) Ratio criterion", fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig1_diagnosis.pdf"))
fig.savefig(os.path.join(HERE, "fig1_diagnosis.png"), dpi=180)
