"""Figures and table values for Section 5.3 (real data).

Data (latifakgun/hope-research):
  argv[1]: runs/v3-modal-l4-2026-10-16/ with t11v3, t11fmv3, t11iobign16v3, t11fmbign16v3 *_FULL_results.json
           (rows as written by notebooks/build_t11_nb.py); t11iobign20v3 / t11fmbign20v3 are used if present.
  argv[2]: runs/noisy_selection_v1/results.json (selection inputs measured with S_sel shots per input).
Primary rows: live, EC-LEARN unit, H-body >= chance + 5 (k6_ok), n = 8-12 (47 rows, 16 units).
  fig_real_shots:  (a) median accuracy minus target vs shots per input, primary rows; shot saving of
                   greedy-residual over H-body per row, (b) selection free, (c) selection inputs measured at
                   1000 shots per input (debiased greedy-residual).
  fig_real_nscale: accuracy vs number of qubits, every row of ionosphere / Fashion-MNIST with H = 2.
  fig_real_acc:    greedy-residual vs H-body, live rows.
  rbf_table (in fig_real_data.json): every kernel against the RBF kernel (Table in Section 5.3).
The full kernel is left out of the shot curves: its S counts shots per kernel entry (overlap test), not per input.
Usage: python make_fig_real.py /path/to/runs/v3-modal-l4-2026-10-16 /path/to/runs/noisy_selection_v1/results.json
Also writes fig_real_data.json (every plotted value) next to the figures.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC, NSEL = sys.argv[1], sys.argv[2]
FILES = ["t11v3_FULL_results.json", "t11fmv3_FULL_results.json",
         "t11iobign16v3_FULL_results.json", "t11fmbign16v3_FULL_results.json",
         "t11iobign20v3_FULL_results.json", "t11fmbign20v3_FULL_results.json"]
R, used = [], []
for f in FILES:
    if os.path.exists(os.path.join(SRC, f)):
        R += json.load(open(os.path.join(SRC, f)))["rows"]; used.append(f)
S_GRID = [1e2, 3e2, 1e3, 3e3, 1e4, 1e5]
X_EXACT = 1e6                      # plotting position of the exact limit
S_SEL = 1000                       # selection shots per input in fig_real_saving (b)
prim = [r for r in R if r["live"] and r["ec"] == "EC-LEARN" and r["k6_ok"] and r["n"] <= 12]
live = [r for r in R if r["live"]]
assert len(prim) == 47, len(prim)

ARMS = [("A7_resid", "greedy-residual", "#c0392b", "o"),
        ("A3_topp", "top-capacity", "#e67e22", "s"),
        ("A2_hbody", "H-body", "#2c6fbb", "^"),
        ("A1_random", "random", "0.55", "v")]
DS = {"breast_cancer": ("breast cancer", "#8e44ad", "o"),
      "fashion_mnist": ("Fashion-MNIST", "#16a085", "s"),
      "ionosphere": ("ionosphere", "#d35400", "^")}

plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
out = {"files": used, "primary_rows": len(prim), "live_rows": len(live), "all_rows": len(R)}

# ---------------- Figure: shots (curves, saving without and with the selection cost) ----------------
fig = plt.figure(figsize=(7.0, 2.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.38)
a1 = fig.add_subplot(gs[0])
xs = S_GRID + [X_EXACT]
out["curves"] = {}
for arm, lab, c, m in ARMS:
    Y = np.array([np.array(r["curve_" + arm]) - r["tau"] for r in prim])     # (47, 7)
    med, q1, q3 = np.median(Y, 0), np.percentile(Y, 25, 0), np.percentile(Y, 75, 0)
    a1.plot(xs, med, marker=m, color=c, ms=3, lw=1.1, label=lab)
    a1.fill_between(xs, q1, q3, color=c, alpha=0.13, lw=0)
    out["curves"][arm] = dict(median=med.tolist(), q1=q1.tolist(), q3=q3.tolist())
a1.axhline(0, color="0.3", lw=0.8, ls="--")
a1.text(2.2e5, -1.9, "target", fontsize=7, color="0.3", ha="right")
a1.set_xscale("log"); a1.set_xticks(xs)
a1.set_xticklabels([r"$10^2$", "", r"$10^3$", "", r"$10^4$", r"$10^5$", "ex."])
a1.set_xlabel("shots per input $N$"); a1.set_ylabel("accuracy $-$ target (points)")
a1.set_ylim(-16, 15); a1.legend(fontsize=6.5, frameon=False, loc="lower right")
a1.set_title("(a) reaching the target", fontsize=8.5, loc="left")

ns_rows = {r["key"] + f"|{r['H']}|{r['p']}": r for r in json.load(open(NSEL))["rows"]}
c1 = fig.add_subplot(gs[1]); c2 = fig.add_subplot(gs[2], sharey=c1)
jit = np.random.default_rng(0)
out["saving"] = {"free": [], "measured": []}
for ax, mode in ((c1, "free"), (c2, "measured")):
    all_ratio = []
    for i, (ds, (lab, c, m)) in enumerate(DS.items()):
        g = [r for r in prim if r["ds"] == ds]
        if mode == "free":
            ratio = np.array([r["S_A2_hbody"] / r["S_A7_resid"] for r in g])
            sstar = np.array([r["S_A7_resid"] for r in g])
        else:
            nr = [ns_rows[r["key"] + f"|{r['H']}|{r['p']}"] for r in g]
            ratio = np.array([x["arms"][f"A7d|{S_SEL}"]["ratio_vs_hbody"] for x in nr])
            sstar = np.array([x["arms"][f"A7d|{S_SEL}"]["S_star"] for x in nr])
        bound = sstar <= 100.0001
        x = i + jit.uniform(-0.17, 0.17, len(g))
        ax.scatter(x[~bound], ratio[~bound], s=12, color=c, marker="o", edgecolor="none", alpha=0.9, zorder=3)
        ax.scatter(x[bound], ratio[bound], s=12, facecolor="white", edgecolor=c, marker="o", lw=0.9, zorder=3)
        for xx, yy in zip(x[bound], ratio[bound]):          # "at least": short arrow upward
            ax.annotate("", xy=(xx, yy * 1.9), xytext=(xx, yy * 1.12),
                        arrowprops=dict(arrowstyle="-|>", color=c, lw=0.7, mutation_scale=5), zorder=2)
        med = float(np.median(ratio))
        ax.plot((i - 0.3, i + 0.3), (med, med), color="k", lw=1.5, zorder=4)
        ax.text(i + 0.32, med, (f"{med:.1f}" if med < 100 else f"{med:.0f}") + "$\\times$",
                fontsize=6.5, va="center", ha="left")
        all_ratio += ratio.tolist()
        out["saving"][mode] += [dict(ds=r["ds"], n=r["n"], p=r["p"], H=r["H"], ent=r["ent"], bw=r["bw"],
                                     ratio=float(q), lower_bound=bool(bb)) for r, q, bb in zip(g, ratio, bound)]
    pooled = float(np.median(all_ratio)); out["saving"][mode + "_median"] = pooled
    ax.axhline(pooled, color="0.35", lw=0.9, ls="--", zorder=1)
    ax.text(2.92, pooled / 1.12, f"all: {pooled:.1f}$\\times$", fontsize=6.5, color="0.3", ha="right", va="top")
    ax.axhline(1, color="0.6", lw=0.9, zorder=1)
    ax.set_yscale("log"); ax.set_ylim(0.6, 1.5e3); ax.set_xlim(-0.5, 2.95)
    ax.set_xticks(range(len(DS)))
    ax.set_xticklabels(["breast\ncancer", "Fashion-\nMNIST", "iono-\nsphere"], fontsize=7)
    ax.set_yticks([1, 10, 100, 1000]); ax.set_yticklabels(["1", "10", "100", "1000"])
plt.setp(c2.get_yticklabels(), visible=False)
c1.set_ylabel("times fewer shots than H-body", fontsize=8)
c1.set_title("(b) selection inputs exact", fontsize=8.5, loc="left")
c2.set_title(f"(c) measured, {S_SEL} shots", fontsize=8.5, loc="left")
fig.subplots_adjust(left=0.085, right=0.985, bottom=0.2, top=0.9)
fig.savefig(os.path.join(HERE, "fig_real_shots.pdf"))
fig.savefig(os.path.join(HERE, "fig_real_shots.png"), dpi=180)

# ---------------- Figure: accuracy vs number of qubits ----------------
# every row (no selection on outcome) of the two datasets run up to large n, H = 2
NARMS = [("A0", "full kernel", "k", "D")] + ARMS
fig, b_axes = plt.subplots(1, 2, figsize=(7.0, 2.7), sharey=True)
out["acc_vs_n"] = {}
for j, (ax, (ds, letter)) in enumerate(zip(b_axes, (("ionosphere", "a"), ("fashion_mnist", "b")))):
    g = [r for r in R if r["ds"] == ds and r["H"] == 2]
    ns = sorted({r["n"] for r in g})
    out["acc_vs_n"][ds] = {"n": ns, "rows_per_n": [sum(1 for r in g if r["n"] == n) for n in ns]}
    for arm, lab, c, m in NARMS:
        y = [float(np.median([r["acc_" + arm] for r in g if r["n"] == n])) for n in ns]
        ax.plot(ns, y, marker=m, color=c, ms=3.2, lw=1.1, label=lab)
        out["acc_vs_n"][ds][arm] = y
    yr = [float(np.median([r["rbf"] for r in g if r["n"] == n])) for n in ns]
    ax.plot(ns, yr, color="0.5", lw=0.9, ls="--", label="RBF (classical)")
    out["acc_vs_n"][ds]["rbf"] = yr
    ch = float(np.median([r["chance"] for r in g]))
    ax.axhline(ch, color="0.75", lw=0.8, ls=":")
    ax.text(ns[-1], ch + 0.8, "chance", fontsize=7, color="0.5", ha="right", va="bottom")
    ax.set_xticks(ns); ax.set_xlabel("qubits $n$")
    ax.set_title(f"({letter}) {DS[ds][0]}", fontsize=9, loc="left")
b_axes[0].set_ylabel("accuracy (%)"); b_axes[0].set_ylim(45, 100)
h, l = b_axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=7, frameon=False, loc="center right")
fig.tight_layout(rect=(0, 0, 0.8, 1))
fig.savefig(os.path.join(HERE, "fig_real_nscale.pdf"))
fig.savefig(os.path.join(HERE, "fig_real_nscale.png"), dpi=180)

# ---------------- Figure: accuracy vs H-body (single panel) ----------------
fig, b1 = plt.subplots(1, 1, figsize=(3.4, 3.0))
out["acc"] = []
for p, c, m in ((16, "#7f8c8d", "o"), (64, "#c0392b", "s")):
    g = [r for r in live if r["p"] == p]
    x = [r["acc_A2_hbody"] for r in g]; y = [r["acc_A7_resid"] for r in g]
    d = np.array(y) - np.array(x)
    b1.scatter(x, y, s=12, color=c, marker=m, edgecolor="none", alpha=0.8,
               label=f"$p = {p}$: {int((d > 0).sum())}/{len(d)} above")
    out["acc"] += [dict(ds=r["ds"], n=r["n"], p=p, H=r["H"], hbody=r["acc_A2_hbody"], greedy=r["acc_A7_resid"],
                        ec=r["ec"]) for r in g]
b1.plot((45, 100), (45, 100), color="0.5", lw=0.8)
b1.set_xlim(45, 100); b1.set_ylim(45, 100)
b1.set_xlabel("H-body accuracy (%)"); b1.set_ylabel("greedy-residual accuracy (%)")
b1.legend(fontsize=7.5, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_real_acc.pdf"))
fig.savefig(os.path.join(HERE, "fig_real_acc.png"), dpi=180)

# ---------------- Table: every kernel against the RBF kernel ----------------
TAB = [("A0", "full kernel"), ("A1_random", "random"), ("A2_hbody", "H-body"),
       ("A3_topp", "top-capacity"), ("A7_resid", "greedy-residual")]
out["rbf_table"] = {}
for setname, g in (("concentrated", [r for r in R if r["ec"] == "EC-LEARN"]), ("all", R)):
    rows = {}
    for arm, lab in TAB:
        q = np.array([r["acc_" + arm] for r in g]); rb = np.array([r["rbf"] for r in g]); d = q - rb
        rows[lab] = dict(n=len(g), median_acc=float(np.median(q)), median_gap=float(np.median(d)),
                         wins=int((d > 0).sum()), smallest_gap=float(d.max()))
    rows["RBF"] = dict(n=len(g), median_acc=float(np.median([r["rbf"] for r in g])))
    out["rbf_table"][setname] = rows

json.dump(out, open(os.path.join(HERE, "fig_real_data.json"), "w"), indent=1)
print("ok", used, {k: v for k, v in out.items() if isinstance(v, int)},
      {k: round(v, 2) for k, v in out["saving"].items() if k.endswith("median")})
