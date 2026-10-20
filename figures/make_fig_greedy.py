"""Figure for Section 4.4: top-capacity vs greedy-residual selection, budget p = 2.

Three terms drawn as vectors of their values over the data; length = sqrt(capacity),
the angle between two vectors encodes how much information they share.
(a, b) redundant case: psi_2 nearly parallel to psi_1. Top-capacity keeps the two
       longest (psi_1, psi_2); greedy-residual keeps psi_1 and then the term with the
       largest component orthogonal to it, psi_3 (0.55 > 0.37).
(c, d) non-redundant case: psi_2 at 45 degrees; its residual 0.65 > 0.55, so both rules agree.
Drawn in the layout and colors of the other figures
(one row of four panels, full width; top-capacity orange, greedy-residual red as in Figures 7-10).
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TOP, GREEDY, OFF = "#e67e22", "#c0392b", "0.65"

P1 = np.array([1.0, 0.0])
P3 = np.array([0.0, 0.55])
cases = {
    "redundant": 0.82 * np.array([np.cos(np.radians(27)), np.sin(np.radians(27))]),
    "nonredundant": 0.92 * np.array([np.cos(np.radians(45)), np.sin(np.radians(45))]),
}


def draw(ax, vecs, kept, color, show_resid):
    ax.axhline(0, color="0.88", lw=0.6, zorder=0)
    ax.axvline(0, color="0.88", lw=0.6, zorder=0)
    for name, v in vecs.items():
        on = name in kept
        ax.annotate("", xy=v, xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color if on else OFF,
                                    lw=2.0 if on else 1.0, shrinkA=0, shrinkB=0,
                                    mutation_scale=9))
        off = {"1": (0.0, -0.11), "2": (0.04, 0.06), "3": (-0.1, 0.04)}[name]
        ax.text(v[0] + off[0], v[1] + off[1], rf"$\psi_{name}$", fontsize=9,
                ha="center", va="center", color="k" if on else "0.45")
    if show_resid:
        q = vecs["2"]
        ax.plot([q[0], q[0]], [0, q[1]], ls=(0, (2, 2)), color=GREEDY if "2" in kept else "0.45", lw=1.0)
        ax.text(q[0] + 0.04, q[1] / 2, f"{q[1]:.2f}", fontsize=7.5, va="center",
                color=GREEDY if "2" in kept else "0.4")
        ax.text(0.04, P3[1] / 2, f"{P3[1]:.2f}", fontsize=7.5, va="center",
                color=GREEDY if "3" in kept else "0.4")
    ax.set_xlim(-0.2, 1.15)
    ax.set_ylim(-0.2, 0.8)
    ax.set_aspect("equal")
    ax.axis("off")


plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.0))
for r, (case, q) in enumerate(cases.items()):
    vecs = {"1": P1, "2": q, "3": P3}
    resid = {k: (v - (v @ P1) * P1)[1] if k != "1" else 1.0 for k, v in vecs.items()}
    greedy_kept = {"1", max(["2", "3"], key=lambda k: resid[k])}
    top_kept = {"1", max(["2", "3"], key=lambda k: np.linalg.norm(vecs[k]))}
    draw(axes[2 * r], vecs, top_kept, TOP, False)
    draw(axes[2 * r + 1], vecs, greedy_kept, GREEDY, True)
    print(case, "top:", sorted(top_kept), "greedy:", sorted(greedy_kept),
          "resid2=%.2f resid3=%.2f" % (resid["2"], resid["3"]))
titles = ["(a) top-capacity", "(b) greedy-residual", "(c) top-capacity", "(d) greedy-residual"]
for ax, t in zip(axes, titles):
    ax.set_title(t, fontsize=9, loc="left")
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.83, wspace=0.12)
for i, lab in ((0, "terms carrying the same information"), (2, "terms carrying different information")):
    l, r = axes[i].get_position(), axes[i + 1].get_position()
    xm = (l.x0 + r.x1) / 2
    fig.text(xm, 0.95, lab, ha="center", va="center", fontsize=9, color="0.25")
    fig.add_artist(plt.Line2D([l.x0 + 0.01, r.x1 - 0.01], [0.905, 0.905], color="0.6", lw=0.8))
xs = (axes[1].get_position().x1 + axes[2].get_position().x0) / 2
fig.add_artist(plt.Line2D([xs, xs], [0.05, 0.92], color="0.85", lw=0.8))
fig.savefig(os.path.join(HERE, "fig_greedy.pdf"))
fig.savefig(os.path.join(HERE, "fig_greedy.png"), dpi=200)
print("ok: fig_greedy.pdf")
