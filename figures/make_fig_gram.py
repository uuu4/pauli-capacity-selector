"""Figure for the Introduction (geometric view): the Gram matrix collapses to the identity.

Fidelity kernel k(x,x') = |<psi(x)|psi(x')>|^2 of the ZZ feature map at the reference
setting (depth 8, eps = 1, bandwidth 0.3), exact statevector, for n = 4, 8, 12, 16.
30 inputs drawn around three random centers (10 each, Gaussian spread 0.1 per coordinate),
ordered by center, so similar inputs form diagonal blocks.
Encoder: src/qk_core.py (encode), loaded from the path given as argv[1].
Usage: python make_fig_gram.py /path/to/hope-research
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ns = dict(np=np, xp=np, GPU=False, CDT=np.complex128, FDT=np.float64,
          to_xp=lambda a, dt=None: np.asarray(a, dtype=dt) if dt else np.asarray(a), to_np=np.asarray)
exec(open(os.path.join(sys.argv[1], "src", "qk_core.py")).read(), ns)

K_CENTRES, PER, SIG, SEED = 3, 10, 0.1, 0
NS = (4, 8, 12, 16)


def gram(n):
    rng = np.random.default_rng(SEED)
    C = rng.uniform(0, 2 * np.pi, (K_CENTRES, n))
    X = np.concatenate([C[i] + SIG * rng.normal(size=(PER, n)) for i in range(K_CENTRES)])
    psi = ns["encode"](X, n, 8, 1.0, 0.3)
    return np.abs(psi.conj() @ psi.T) ** 2


lab = np.repeat(np.arange(K_CENTRES), PER)
same = (lab[:, None] == lab[None, :]) & ~np.eye(len(lab), dtype=bool)
diff = lab[:, None] != lab[None, :]

plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
fig, axes = plt.subplots(1, len(NS), figsize=(7.0, 2.05))
for ax, n in zip(axes, NS):
    K = gram(n)
    assert np.allclose(np.diag(K), 1.0)
    im = ax.imshow(K, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(rf"$n = {n}$", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.5); s.set_color("0.6")
    o = K[diff].mean()
    ax.set_xlabel(f"same center: {K[same].mean():.2f}\nother centers: " + (f"{o:.3f}" if o >= 1e-3 else "<0.001"),
                  fontsize=7.5, color="0.3")
    print(f"n={n}: within-center mean {K[same].mean():.4f}, across {K[diff].mean():.5f}")
cb = fig.colorbar(im, ax=list(axes), fraction=0.025, pad=0.02)
cb.set_label(r"$k(x, x')$", fontsize=8)
cb.ax.tick_params(labelsize=7)
fig.savefig(os.path.join(HERE, "fig_gram.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(HERE, "fig_gram.png"), dpi=200, bbox_inches="tight")
print("ok: fig_gram.pdf")
