"""Figure for Section 4.1: gate contributions do not add up at the output.

One qubit, |0> -> RY(x) -> H -> RY(2x) -> measure <Z>.
(a) circuit; (b) <Z>(x) for the full circuit and with each gate removed.
Curves are computed by exact 2x2 statevector simulation and checked against
the closed forms: full -sin x, no H cos 3x, no RY(2x) sin x, no RY(x) -sin 2x.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
Z = np.diag([1.0, -1.0])


def ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]])


def expz(x, gates):
    psi = np.array([1.0, 0.0])
    for g in gates:
        psi = g(x) @ psi
    return float(psi @ Z @ psi)


G1, G2, G3 = (lambda x: ry(x)), (lambda x: H), (lambda x: ry(2 * x))
xs = np.linspace(0, 2 * np.pi, 400)
curves = {
    "full circuit": ([G1, G2, G3], lambda x: -np.sin(x)),
    r"without $H$: $R_Y(3x)$": ([G1, G3], lambda x: np.cos(3 * x)),
    r"without $R_Y(2x)$": ([G1, G2], lambda x: np.sin(x)),
    r"without $R_Y(x)$": ([G2, G3], lambda x: -np.sin(2 * x)),
}
vals = {}
for name, (gates, closed) in curves.items():
    v = np.array([expz(x, gates) for x in xs])
    assert np.allclose(v, closed(xs), atol=1e-12), name
    vals[name] = v

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.linewidth": 0.8, "mathtext.fontset": "cm"})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [1.25, 1]})

# (a) circuit
ax1.set_xlim(0, 10); ax1.set_ylim(0, 2); ax1.axis("off")
ax1.plot([0.9, 8.6], [1, 1], color="0.2", lw=1, zorder=1)
ax1.text(0.75, 1, r"$|0\rangle$", ha="right", va="center", fontsize=10)
for xc, lab in [(2.4, r"$R_Y(x)$"), (4.75, r"$H$"), (7.1, r"$R_Y(2x)$")]:
    w = 1.5 if "R_Y" in lab else 0.8
    ax1.add_patch(FancyBboxPatch((xc - w / 2, 0.62), w, 0.76, boxstyle="round,pad=0.02",
                                 fc="white", ec="0.2", lw=1, zorder=2))
    ax1.text(xc, 1, lab, ha="center", va="center", fontsize=10, zorder=3)
ax1.add_patch(FancyBboxPatch((8.6, 0.62), 1.1, 0.76, boxstyle="round,pad=0.02",
                             fc="0.93", ec="0.2", lw=1, zorder=2))
ax1.text(9.15, 1, r"$\langle Z\rangle$", ha="center", va="center", fontsize=10, zorder=3)
ax1.set_title("(a)", loc="left", fontsize=9)

# (b) <Z>(x)
styles = [("k", "-", 1.6), ("C3", "--", 1.1), ("C0", ":", 1.1), ("C2", "-.", 1.1)]
for (name, v), (c, ls, lw) in zip(vals.items(), styles):
    ax2.plot(xs, v, color=c, ls=ls, lw=lw, label=name)
ax2.set_xlim(0, 2 * np.pi); ax2.set_ylim(-1.15, 1.15)
ax2.set_xticks([0, np.pi, 2 * np.pi]); ax2.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
ax2.set_xlabel(r"$x$"); ax2.set_ylabel(r"$\langle Z\rangle$")
ax2.axhline(0, color="0.8", lw=0.6)
ax2.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, handlelength=2.2)
ax2.set_title("(b)", loc="left", fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_gates.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(HERE, "fig_gates.png"), dpi=200, bbox_inches="tight")
print("ok: fig_gates.pdf")
