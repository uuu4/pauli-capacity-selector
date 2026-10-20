"""Overview figure: encoding, full kernel collapse, capacity ranking on unlabeled data, selected kernel.

ZZ feature map at depth 8, eps = 1, bandwidth 0.3, n = 12, exact statevector (as Figure 1 and Section 3).
Gram matrices on 30 inputs drawn around three random centers (10 each, Gaussian spread 0.1), ordered by center.
Capacities of all 630 non-constant Pauli terms of weight <= 2 on 400 unlabeled inputs drawn uniformly;
the 16 terms of largest capacity build the selected kernel (Eq. kS).
Encoder and Pauli expectation values: latifakgun/hope-research src/qk_core.py (encode, pool_layout,
pauli_expect_fast), loaded from the path given as argv[1].
Usage: python make_fig_schematic.py /path/to/hope-research
Writes fig_schematic.pdf/.png and fig_schematic_data.json.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ns = dict(np=np, xp=np, GPU=False, CDT=np.complex128, FDT=np.float64,
          to_xp=lambda a, dt=None: np.asarray(a, dtype=dt) if dt else np.asarray(a), to_np=np.asarray)
exec(open(os.path.join(sys.argv[1], "src", "qk_core.py")).read(), ns)

N, H, P, DEPTH, EPS, BW = 12, 2, 16, 8, 1.0, 0.3
rng = np.random.default_rng(0)
C = rng.uniform(0, 2 * np.pi, (3, N))
X = np.concatenate([C[i] + 0.1 * rng.normal(size=(10, N)) for i in range(3)])
Xs = rng.uniform(0, 2 * np.pi, (400, N))
lay = ns["pool_layout"](N, H); w = lay[3]
psi = ns["encode"](X, N, DEPTH, EPS, BW); psis = ns["encode"](Xs, N, DEPTH, EPS, BW)
E = ns["pauli_expect_fast"](psi, N, lay); Es = ns["pauli_expect_fast"](psis, N, lay)
keep = w >= 1                                   # constant term I excluded, as in all selection rules
E, Es, w = E[:, keep], Es[:, keep], w[keep]
gam = (Es ** 2).mean(0) / 2 ** N                # capacity on the psi scale, psi = <P>/sqrt(2^n)
order = np.argsort(gam)[::-1]; top = order[:P]
Kf = np.abs(psi.conj() @ psi.T) ** 2
F = E[:, top]; Ks = (F @ F.T) / np.sqrt(np.outer((F ** 2).sum(1), (F ** 2).sum(1)))
lab = np.repeat(np.arange(3), 10)
same = (lab[:, None] == lab[None, :]) & ~np.eye(30, dtype=bool); diff = lab[:, None] != lab[None, :]
stats = dict(full_same=float(Kf[same].mean()), full_diff=float(Kf[diff].mean()),
             sel_same=float(Ks[same].mean()), sel_diff=float(Ks[diff].mean()),
             pool=int(len(w)), p=P, top_weights=np.bincount(w[top], minlength=3).tolist(),
             top_capacity_share=float(gam[top].sum() / gam.sum()))

plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
fig = plt.figure(figsize=(7.0, 2.45))
gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1, 1.15, 1], wspace=0.5)
a0, a1, a2, a3 = (fig.add_subplot(gs[i]) for i in range(4))

# (a) encoding and Pauli expansion, drawn as boxes
a0.axis("off"); a0.set_xlim(0, 1); a0.set_ylim(0, 1)
def box(ax, x, y, wd, ht, txt, fc, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), wd, ht, boxstyle="round,pad=0.02", fc=fc, ec="0.35", lw=0.8))
    ax.text(x + wd / 2, y + ht / 2, txt, ha="center", va="center", fontsize=fs)
box(a0, 0.05, 0.74, 0.9, 0.17, "input $x \\in [0, 2\\pi)^n$", "#f2f2f2")
box(a0, 0.05, 0.43, 0.9, 0.2, "circuit $U(x)$\n$\\to |\\psi(x)\\rangle$", "#e8eef8")
box(a0, 0.05, 0.04, 0.9, 0.28, "$k(x,x') = \\sum_P \\psi_P(x)\\,\\psi_P(x')$\n$4^n$ Pauli terms", "#fdf2e3", fs=7.2)
for y0, y1 in ((0.74, 0.65), (0.43, 0.34)):
    a0.annotate("", xy=(0.5, y1), xytext=(0.5, y0), arrowprops=dict(arrowstyle="-|>", color="0.35", lw=0.9))
a0.set_title("(a) encode, expand", fontsize=9, loc="left")

for ax, K, ttl in ((a1, Kf, "(b) all $4^n$ terms"), (a3, Ks, f"(d) top {P} terms")):
    ax.imshow(K, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(ttl, fontsize=9, loc="left")
a1.set_xlabel("Gram $\\approx$ identity", fontsize=8)
a3.set_xlabel("clusters return", fontsize=8)

# (c) capacity ranking on unlabeled data
r = np.arange(1, len(gam) + 1)
a2.semilogy(r, gam[order], color="0.6", lw=1.0)
a2.semilogy(r[:P], gam[order][:P], "o", color="#c0392b", ms=2.8)
a2.set_xscale("log")
a2.set_xlabel("rank of term", fontsize=8); a2.set_ylabel("capacity $\\gamma_i$", fontsize=8, labelpad=1)
a2.tick_params(labelsize=7)
a2.text(0.06, 0.3, f"kept: top {P}\nno labels used", transform=a2.transAxes, ha="left", va="top",
        fontsize=7.5, color="#c0392b")
a2.set_title("(c) rank by capacity", fontsize=9, loc="left")

fig.subplots_adjust(left=0.01, right=0.975, bottom=0.17, top=0.86)
# arrows between panels (figure coordinates)
for axl, axr in ((a0, a1), (a1, a2), (a2, a3)):
    pl, pr = axl.get_position(), axr.get_position()
    fig.patches.append(matplotlib.patches.FancyArrowPatch(
        (pl.x1 + 0.006, 0.5), (pr.x0 - (0.075 if axr is a2 else 0.006), 0.5), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=9, color="0.35", lw=0.9))
fig.savefig(os.path.join(HERE, "fig_schematic.pdf")); fig.savefig(os.path.join(HERE, "fig_schematic.png"), dpi=180)
json.dump(stats, open(os.path.join(HERE, "fig_schematic_data.json"), "w"), indent=1)
print(stats)
