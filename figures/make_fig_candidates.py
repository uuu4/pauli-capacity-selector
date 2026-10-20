"""Candidate figures (to be filtered): real data, selection cost, synthetic survival.

Data (this repository):
  argv[1]: runs/v3-modal-l4-2026-10-16/ (t11v3, t11fmv3, t11iobign16v3, t11fmbign16v3 *_FULL_results.json)
  argv[2]: runs/noisy_selection_v1/results.json
  argv[3]: runs/ folder holding phase_{b,c,c2,d}v3_full_v1/*_FULL_results.json (synthetic)
Writes cand_*.pdf/.png and cand_data.json (every plotted value) next to this script.
Usage: python make_fig_candidates.py RUNS/v3-modal-l4-2026-10-16 RUNS/noisy_selection_v1/results.json RUNS
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SRC, NSEL, SYN = sys.argv[1], sys.argv[2], sys.argv[3]
R, EC = [], {}
for f in ["t11v3_FULL_results.json", "t11fmv3_FULL_results.json",
          "t11iobign16v3_FULL_results.json", "t11fmbign16v3_FULL_results.json",
          "t11iobign20v3_FULL_results.json", "t11fmbign20v3_FULL_results.json"]:
    p = os.path.join(SRC, f)
    if os.path.exists(p):
        d = json.load(open(p)); R += d["rows"]; EC.update(d["ec"])
prim = [r for r in R if r["live"] and r["ec"] == "EC-LEARN" and r["k6_ok"] and r["n"] <= 12]
live = [r for r in R if r["live"]]
DS = {"breast_cancer": ("breast cancer", "#8e44ad", "o"),
      "fashion_mnist": ("Fashion-MNIST", "#16a085", "s"),
      "ionosphere": ("ionosphere", "#d35400", "^")}
COL = {"A0": "k", "A7_resid": "#c0392b", "A3_topp": "#e67e22", "A2_hbody": "#2c6fbb",
       "A1_random": "0.55", "A5_super": "#7d3c98"}
LAB = {"A0": "full kernel", "A7_resid": "greedy-residual", "A3_topp": "top-capacity",
       "A2_hbody": "H-body", "A1_random": "random", "A5_super": "label-aware selector"}
plt.rcParams.update({"font.family": "serif", "font.size": 9, "mathtext.fontset": "cm"})
out = {}


def save(fig, name):
    fig.savefig(os.path.join(HERE, name + ".pdf")); fig.savefig(os.path.join(HERE, name + ".png"), dpi=170)
    plt.close(fig)


# C1: advantage over the full kernel against the concentration of the full kernel, one point per unit
fig, ax = plt.subplots(figsize=(3.6, 3.0))
out["c1"] = []
for ds, (lab, c, m) in DS.items():
    xs, ys = [], []
    for k in sorted({r["key"] for r in R if r["ds"] == ds}):
        g = [r for r in R if r["key"] == k and r["p"] == 64]
        x = EC[k]["effrank_frac"]; y = float(np.median([r["acc_A7_resid"] - r["acc_A0"] for r in g]))
        xs.append(x); ys.append(y)
        out["c1"].append(dict(unit=k, effrank_frac=x, gain=y, ec=EC[k]["status"]))
    ax.scatter(xs, ys, s=22, color=c, marker=m, edgecolor="none", alpha=0.85, label=lab)
ax.axhline(0, color="0.5", lw=0.8)
ax.set_xlabel("effective rank of the full kernel / $M$")
ax.set_ylabel("greedy-residual $-$ full kernel (points)")
ax.legend(fontsize=7.5, frameon=False, loc="upper left")
ax.set_title("selection wins where the full kernel concentrates", fontsize=8.5, loc="left")
fig.tight_layout(); save(fig, "cand_gain_vs_concentration")

# C2: label-free rules against the label-aware selector with the same budget
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True)
out["c2"] = {}
for ax, p in zip(axes, (16, 64)):
    g = [r for r in live if r["p"] == p]
    for a, m in (("A7_resid", "o"), ("A2_hbody", "^")):
        x = [r["acc_A5_super"] for r in g]; y = [r["acc_" + a] for r in g]
        d = np.array(y) - np.array(x)
        ax.scatter(x, y, s=12, color=COL[a], marker=m, edgecolor="none", alpha=0.8,
                   label=f"{LAB[a]}: {int((d >= 0).sum())}/{len(d)} at least as good")
        out["c2"][f"{a}|{p}"] = dict(median_diff=float(np.median(d)), at_least=int((d >= 0).sum()), n=len(d))
    ax.plot((45, 100), (45, 100), color="0.5", lw=0.8)
    ax.set_xlim(45, 100); ax.set_ylim(45, 100)
    ax.set_xlabel("label-aware selector (%)"); ax.set_title(f"$p = {p}$, live rows", fontsize=9, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
axes[0].set_ylabel("label-free rule (%)")
fig.tight_layout(); save(fig, "cand_vs_labelaware")

# C3: shot curves per dataset
S_GRID = [1e2, 3e2, 1e3, 3e3, 1e4, 1e5]; xs = S_GRID + [1e6]
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharey=True)
out["c3"] = {}
for ax, (ds, (lab, c, m)) in zip(axes, DS.items()):
    g = [r for r in prim if r["ds"] == ds]
    for a in ("A7_resid", "A3_topp", "A2_hbody", "A1_random"):
        Y = np.array([np.array(r["curve_" + a]) - r["tau"] for r in g]); med = np.median(Y, 0)
        ax.plot(xs, med, color=COL[a], lw=1.1, marker="o", ms=2.5, label=LAB[a])
        out["c3"][f"{ds}|{a}"] = med.tolist()
    ax.axhline(0, color="0.3", lw=0.8, ls="--"); ax.set_xscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([r"$10^2$", "", r"$10^3$", "", r"$10^4$", r"$10^5$", "ex."])
    ax.set_title(f"{lab} ({len(g)} rows)", fontsize=9, loc="left"); ax.set_xlabel("shots per input")
axes[0].set_ylabel("accuracy $-$ target"); axes[0].set_ylim(-20, 20)
axes[2].legend(fontsize=6.5, frameon=False, loc="lower right")
fig.tight_layout(); save(fig, "cand_curves_by_dataset")


# C4: why the saving is large: accuracy margin above the target vs saving
def sstar(curve, tau):
    c = np.array(curve[:6]); G = np.array(S_GRID)
    if c[0] >= tau: return 1e2
    for i in range(1, 6):
        if c[i] >= tau:
            t = (tau - c[i - 1]) / (c[i] - c[i - 1]); return float(np.exp(np.log(G[i - 1]) + t * np.log(G[i] / G[i - 1])))
    return np.inf


fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
out["c4"] = []
for ds, (lab, c, m) in DS.items():
    g = [r for r in prim if r["ds"] == ds]
    mg = [r["acc_A7_resid"] - r["tau"] for r in g]; rt = [r["S_A2_hbody"] / r["S_A7_resid"] for r in g]
    own = [sstar(r["curve_A2_hbody"], r["acc_A2_hbody"] - 1) / sstar(r["curve_A7_resid"], r["acc_A7_resid"] - 1) for r in g]
    a1.scatter(mg, rt, s=16, color=c, marker=m, edgecolor="none", alpha=0.85, label=lab)
    a2.scatter(np.full(len(own), list(DS).index(ds)) + np.random.default_rng(1).uniform(-0.15, 0.15, len(own)),
               own, s=16, color=c, marker=m, edgecolor="none", alpha=0.85)
    out["c4"] += [dict(ds=ds, margin=a, saving=b, own_target_saving=o) for a, b, o in zip(mg, rt, own)]
a1.set_yscale("log"); a1.set_xlabel("greedy-residual accuracy above the target (points)")
a1.set_ylabel("times fewer shots than H-body"); a1.legend(fontsize=7, frameon=False)
a1.set_title("(a) common target", fontsize=9, loc="left")
a2.set_yscale("log"); a2.axhline(1, color="0.5", lw=0.8)
a2.set_xticks(range(3)); a2.set_xticklabels([v[0] for v in DS.values()], fontsize=8)
a2.set_title("(b) each rule's own target (own accuracy $-$ 1)", fontsize=9, loc="left")
fig.tight_layout(); save(fig, "cand_saving_decomposition")

# C5/C6: selection from measured inputs: saving and accuracy against the selection shots
NR = json.load(open(NSEL))["rows"]; SS = [100, 300, 1000, 3000]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
out["c5"] = {}
for arm, lab, c, ls in (("A7d", "greedy-residual, corrected", "#c0392b", "-"), ("A7", "greedy-residual, raw", "#c0392b", ":"),
                        ("A3d", "top-capacity, corrected", "#e67e22", "-"), ("A3", "top-capacity, raw", "#e67e22", ":")):
    med = [float(np.median([x["arms"][f"{arm}|{s}"]["ratio_vs_hbody"] for x in NR])) for s in SS]
    reach = [sum(1 for x in NR if np.isfinite(x["arms"][f"{arm}|{s}"]["S_star"] if x["arms"][f"{arm}|{s}"]["S_star"] is not None else np.inf)) for s in SS]
    loss = [float(np.median([x["arms"][f"{arm}|{s}"]["curve"][-1] - x["arms"][f"{arm[:2]}|inf"]["curve"][-1] for x in NR])) for s in SS]
    a1.plot(SS, med, color=c, ls=ls, marker="o", ms=3, label=lab); a2.plot(SS, loss, color=c, ls=ls, marker="o", ms=3, label=lab)
    out["c5"][arm] = dict(median_saving=med, rows_reaching=reach, median_accuracy_change=loss)
a1.set_xscale("log"); a1.set_xticks(SS); a1.set_xticklabels([str(s) for s in SS])
a1.set_xlabel("selection shots per input $S_{\\mathrm{sel}}$"); a1.set_ylabel("median times fewer shots than H-body")
a1.axhline(1, color="0.6", lw=0.8); a1.legend(fontsize=6.8, frameon=False, loc="upper left")
a1.set_title("(a) saving with the selection cost", fontsize=9, loc="left")
a2.set_xscale("log"); a2.set_xticks(SS); a2.set_xticklabels([str(s) for s in SS])
a2.set_xlabel("selection shots per input $S_{\\mathrm{sel}}$"); a2.set_ylabel("noise-free accuracy change (points)")
a2.axhline(0, color="0.6", lw=0.8); a2.set_title("(b) accuracy lost to selection noise", fontsize=9, loc="left")
fig.tight_layout(); save(fig, "cand_selection_noise")

# C7: map of greedy-residual minus H-body (p = 64) over the real-data grid
fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
out["c7"] = {}
cols = [(0.5, 0.15), (0.5, 0.25), (1.0, 0.15), (1.0, 0.25)]
for ax, (ds, (lab, c, m)) in zip(axes, DS.items()):
    ns = sorted({r["n"] for r in R if r["ds"] == ds})
    M = np.full((len(ns), len(cols)), np.nan)
    for i, n in enumerate(ns):
        for j, (e, bw) in enumerate(cols):
            g = [r for r in R if r["ds"] == ds and r["n"] == n and r["ent"] == e and r["bw"] == bw and r["p"] == 64]
            if g: M[i, j] = np.median([r["acc_A7_resid"] - r["acc_A2_hbody"] for r in g])
    im = ax.imshow(M, cmap="RdBu", vmin=-25, vmax=25, aspect="auto", origin="lower")
    for i in range(len(ns)):
        for j in range(len(cols)):
            if np.isfinite(M[i, j]): ax.text(j, i, f"{M[i, j]:+.0f}", ha="center", va="center", fontsize=7)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([f"$\\varepsilon$={e}\n$c$={bw}" for e, bw in cols], fontsize=6.5)
    ax.set_yticks(range(len(ns))); ax.set_yticklabels(ns); ax.set_title(lab, fontsize=9, loc="left")
    out["c7"][ds] = dict(n=ns, cols=cols, values=M.tolist())
axes[0].set_ylabel("qubits $n$")
fig.colorbar(im, ax=axes, fraction=0.025, label="greedy $-$ H-body")
save(fig, "cand_map_vs_hbody")

# C8: synthetic, rows where the full kernel is dead but the labels are learnable: which rule brings it back
PH = [("phase_bv3_full_v1/phase_bv3_FULL_results.json", "1"), ("phase_cv3_full_v1/phase_cv3_FULL_results.json", "2"),
      ("phase_c2v3_full_v1/phase_c2v3_FULL_results.json", "3"), ("phase_dv3_full_v1/phase_dv3_FULL_results.json", "4")]
fig, ax = plt.subplots(figsize=(4.6, 2.8))
out["c8"] = {}
W = 0.2
for k, (f, nm) in enumerate(PH):
    rows = [r for r in json.load(open(os.path.join(SYN, f)))["rows"]
            if r["label"] != "qlocal" and r["A0"] < 53 and r["A5_super"] >= 55]
    for j, a in enumerate(("A3_topp", "A7_resid", "A2_hbody", "A1_random")):
        frac = 100 * np.mean([r[a] >= 55 for r in rows])
        ax.bar(k + (j - 1.5) * W, frac, W, color=COL[a], label=LAB[a] if k == 0 else None)
        out["c8"][f"{nm}|{a}"] = dict(alive=int(sum(r[a] >= 55 for r in rows)), rows=len(rows))
ax.set_xticks(range(4)); ax.set_xticklabels([f"experiment {nm}" for _, nm in PH], fontsize=8)
ax.set_ylabel("rows brought back to life (%)")
ax.legend(fontsize=7, frameon=False, loc="upper left")
ax.set_title("synthetic: full kernel dead, labels learnable", fontsize=8.5, loc="left")
fig.tight_layout(); save(fig, "cand_synthetic_revival")

json.dump(out, open(os.path.join(HERE, "cand_data.json"), "w"), indent=1, default=float)
print("ok", sorted(out))

# C9: budget sweep on synthetic data, rows where the full kernel is dead and the labels are learnable
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.5), sharey=True)
out["c9"] = {}
for ax, (f, nm) in zip(axes, PH):
    rows = [r for r in json.load(open(os.path.join(SYN, f)))["rows"] if r["label"] != "qlocal"]
    ps = sorted({r["p"] for r in rows}); pmax = ps[-1]
    grp = {}
    for r in rows:
        grp.setdefault((r["n"], r["depth"], r["ent"], r.get("bw", 0.3), r.get("rep", 1), r.get("jitter", 0.0),
                        r["label"], r["H"]), {})[r["p"]] = r
    keep = [g for g in grp.values() if pmax in g and g[pmax]["A0"] < 53 and g[pmax]["A5_super"] >= 55]
    for a in ("A7_resid", "A3_topp", "A5_super", "A2_hbody", "A1_random"):
        y = [float(np.median([g[p][a] for g in keep if p in g])) for p in ps]
        ax.plot(ps, y, marker="o", ms=3, lw=1.1, color=COL[a], label=LAB[a])
        out["c9"][f"{nm}|{a}"] = dict(p=ps, median=y, groups=len(keep))
    a0 = float(np.median([g[pmax]["A0"] for g in keep]))
    ax.axhline(a0, color="k", ls="--", lw=0.9, label="full kernel"); out["c9"][f"{nm}|A0"] = a0
    ax.axhline(50, color="0.75", ls=":", lw=0.8)
    ax.set_xscale("log", base=2); ax.set_xticks(ps); ax.set_xticklabels([str(p) for p in ps], fontsize=7.5)
    ax.set_xlabel("budget $p$"); ax.set_title(f"experiment {nm} ({len(keep)})", fontsize=8.5, loc="left")
axes[0].set_ylabel("accuracy (%)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, fontsize=7, frameon=False, loc="lower center", ncol=6)
fig.tight_layout(rect=(0, 0.08, 1, 1)); save(fig, "cand_budget_sweep")

# C10: boxplots per dataset and number of qubits, every row, split by whether the full kernel concentrates
NC = {"breast_cancer": (8, 10, 12), "fashion_mnist": (8, 10, 12, 16, 20), "ionosphere": (8, 10, 12, 16, 20)}
fig, axes = plt.subplots(3, 5, figsize=(7.0, 6.6), sharey=True)
for ax in axes[0, 3:]: ax.set_visible(False)           # breast cancer runs only up to n = 12
out["c10"] = {}
BX = [("A0", "full"), ("A1_random", "random"), ("A2_hbody", "H-body"), ("A3_topp", "top-cap."),
      ("A7_resid", "greedy"), ("rbf", "RBF")]
for i, (ds, (lab, c, m)) in enumerate(DS.items()):
    for j, n in enumerate(NC[ds]):
        ax = axes[i, j]
        g = [r for r in R if r["ds"] == ds and r["n"] == n]
        ax.set_title(f"{lab}\n$n = {n}$" + (f" ({len(g)} rows)" if g else ""), fontsize=7, loc="left")
        ax.set_xticks(range(len(BX))); ax.set_xticklabels([b_ for _, b_ in BX], rotation=60, fontsize=6)
        if not g:
            ax.text(0.5, 0.5, "running", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="0.5")
            continue
        for jj, (a, short) in enumerate(BX):
            for k, st in enumerate(("concentrated", "NO-EC")):   # concentrated: every class but NO-EC (EC-LEARN, EC-BOTH)
                v = [r["rbf"] if a == "rbf" else r["acc_" + a] for r in g if (r["ec"] != "NO-EC") == (k == 0)]
                if not v: continue
                col = "0.5" if a == "rbf" else ("#1f3a93" if a == "A0" else COL[a])
                bp = ax.boxplot(v, positions=[jj + (k - 0.5) * 0.36], widths=0.32, patch_artist=True,
                                showfliers=False, medianprops=dict(color="k", lw=1), manage_ticks=False)
                bp["boxes"][0].set(facecolor=col, alpha=0.85 if k == 0 else 0.3, edgecolor=col)
                for el in bp["whiskers"] + bp["caps"]: el.set(color=col)
                out["c10"][f"{ds}|{n}|{a}|{st}"] = dict(n=len(v), median=float(np.median(v)),
                                                       q1=float(np.percentile(v, 25)), q3=float(np.percentile(v, 75)))
        ch = float(np.median([r["chance"] for r in g])); ax.axhline(ch, color="0.75", ls=":", lw=0.8)
        ax.set_xlim(-0.6, len(BX) - 0.4)
        ax.set_xticks(range(len(BX))); ax.set_xticklabels([b_ for _, b_ in BX], rotation=60, fontsize=6)
for i in range(3): axes[i, 0].set_ylabel("test accuracy (%)")
fig.tight_layout(); save(fig, "cand_boxplots")
json.dump(out, open(os.path.join(HERE, "cand_data.json"), "w"), indent=1, default=float)
print("ok2", {k: v for k, v in out.get("c9", {}).items() if k.endswith("A0")})
