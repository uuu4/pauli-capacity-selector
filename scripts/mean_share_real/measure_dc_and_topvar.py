"""HOPE v3 — #15 & #16 Dogrulama Betigi (sklearn >= 1.6 uyumlu, H=2 ve H=4 dahil):
1) B1: Agirlik sinifi basina toplam ve terim-basi kapasite (seyrelme / dilution)
2) B3: X, Y, Z (w=1) ve w=2 icin gamma = mu^2 + Var ayrisimi
3) Ongoru A vs B: Top-Variance (Var_x[psi_i]) vs Top-Capacity (E_x[psi_i^2]) vs Greedy-Residual (A7)
   gercek KRR test dogrulugu (ayni 30 split, H in {2, 4}, p in {16, 64})
4) Sabit vektor 1'in secilen alt uzaydaki izdusum orani: ||Pi_{S_t} 1||^2 / ||1||^2 (t=1..16)
5) #16 Madde 4: G5 (S_sel=1000 ve S_sel=S*) icin satir ve birim (cluster) bootstrap GA
"""
import json, os, pathlib, time
import numpy as np
import sklearn

HOPE = pathlib.Path(__file__).resolve().parents[2]   # repository root
G = dict(np=np, xp=np, GPU=False, CDT=np.complex128, FDT=np.float64,
         to_xp=lambda a, dtype=None: np.asarray(a, dtype=dtype),
         to_np=np.asarray)
for _f in ("qk_core.py", "phaseb_core.py", "gate4_core.py", "t11_core.py"):
    exec(compile((HOPE / "src" / _f).read_text(encoding="utf-8"), _f, "exec"), G)

# Gecici dizindeki fashion-mnist_train.csv'yi de bulabilmesi icin _find_local saricisi
_orig_find_local = G["_find_local"]
def _find_local_ext(name, roots=(".", "/kaggle/input", "/data")):
    return _orig_find_local(name, roots=roots)
G["_find_local"] = _find_local_ext


def max_flips_for(n):
    return None if n >= 16 else {8: 200, 10: 120, 12: 120, 14: 60}.get(n, 200)


def eval_arm_krr(Et, Ee, yt, ye, idx):
    """Secilen indeks kumesi idx icin makaledeki Eq. kS (birim kosegen normalize) KRR dogrulugu."""
    Ft, Fe = Et[:, idx], Ee[:, idx]
    Kt = G["kernel_pair"](Ft, Ft)
    Ke = G["kernel_pair"](Fe, Ft)
    accs = G["krr_multi"](Kt, Ke, {"y": yt}, {"y": ye})
    return float(accs["y"][0])


def one_projection_curve(E, idx, max_t=16):
    """Adim t=1..max_t icin ||Pi_{S_t} 1||^2 / ||1||^2."""
    M = E.shape[0]
    ones = np.ones(M, dtype=float)
    ones_norm2 = float(M)
    curve = []
    for t in range(1, min(len(idx), max_t) + 1):
        A = E[:, idx[:t]]
        Q, _ = np.linalg.qr(A, mode="reduced")
        proj2 = float(np.sum((Q.T @ ones) ** 2))
        curve.append(proj2 / ones_norm2)
    return curve


def run_unit_analysis(ds_name, n, d, ent, bw, n_splits=30, seed0=20260916):
    X_raw, y_all = G["load_real"](ds_name)
    splits = G["stratified_splits"](y_all, 30, seed0, 0.7)[:n_splits]
    # H=2 havuzu (keep_low=2 oldugu icin H=4 havuzunun w<=2 alt kumesiyle BIREBIR aynidir)
    layout = G["pool_layout"](n, 2)
    fs, blocks, idx_all, ww = layout
    w_pool = np.asarray(ww)

    mi_h2 = np.where((w_pool <= 2) & (w_pool >= 1))[0]
    idx_h2 = idx_all[mi_h2]
    w_h2 = w_pool[mi_h2]
    mask_x1 = np.zeros(len(w_h2), dtype=bool)
    mask_y1 = np.zeros(len(w_h2), dtype=bool)
    mask_z1 = np.zeros(len(w_h2), dtype=bool)
    for k in range(n):
        dk = (idx_h2 // (4 ** k)) % 4
        mask_x1 |= (w_h2 == 1) & (dk == 1)
        mask_y1 |= (w_h2 == 1) & (dk == 2)
        mask_z1 |= (w_h2 == 1) & (dk == 3)
    mask_w1 = w_h2 == 1
    mask_w2 = w_h2 == 2

    stats_decomp = {k: {"gamma": [], "mu2": [], "var": []} for k in ("X1", "Y1", "Z1", "w1", "w2")}
    tot_w1, tot_w2 = [], []
    proj_a3, proj_a7, proj_var = [], [], []
    w_sel = {(H, p): {"A3_topp": [], "A3_topvar": [], "A7_resid": [], "A7_resid_c": []}
             for H in (2,) for p in (16, 64)}
    accs = {(H, p): {"A3_topp": [], "A3_topvar": [], "A7_resid": [], "A7_resid_c": []}
            for H in (2,) for p in (16, 64)}

    for s in range(n_splits):
        tr, te = splits[s]
        Xt, Xe = G["preprocess"](X_raw[tr], X_raw[te], n, seed=seed0)
        yt, ye = y_all[tr], y_all[te]
        psi_tr = G["encode"](Xt, n, d, ent, bw, norm="none")
        psi_te = G["encode"](Xe, n, d, ent, bw, norm="none")
        Et_all = G["pauli_expect_fast"](psi_tr, n, layout)
        Ee_all = G["pauli_expect_fast"](psi_te, n, layout)

        Et_h2 = Et_all[:, mi_h2]
        gam2 = np.mean(Et_h2 ** 2, axis=0)
        mu2_2 = (np.mean(Et_h2, axis=0)) ** 2
        var2 = np.var(Et_h2, axis=0)
        tot_w1.append(float(np.sum(gam2[mask_w1])))
        tot_w2.append(float(np.sum(gam2[mask_w2])))
        for k, m in [("X1", mask_x1), ("Y1", mask_y1), ("Z1", mask_z1), ("w1", mask_w1), ("w2", mask_w2)]:
            stats_decomp[k]["gamma"].append(float(np.mean(gam2[m])))
            stats_decomp[k]["mu2"].append(float(np.mean(mu2_2[m])))
            stats_decomp[k]["var"].append(float(np.mean(var2[m])))

        for H in (2,):
            mi = np.where((w_pool <= H) & (w_pool >= 1))[0]
            wH = w_pool[mi]
            Et = Et_all[:, mi]
            Ee = Ee_all[:, mi]
            gam = G["capacity"](Et)
            var = np.var(Et, axis=0)
            Et_c = Et - np.mean(Et, axis=0, keepdims=True)

            for p in (16, 64):
                idx_a3 = np.asarray(G["select_topp"](gam, p))
                idx_var = np.asarray(G["select_topp"](var, p))
                idx_a7 = np.asarray(G["select_greedy_residual"](Et, p))
                idx_a7c = np.asarray(G["select_greedy_residual"](Et_c, p))

                for name, idx in [("A3_topp", idx_a3), ("A3_topvar", idx_var),
                                  ("A7_resid", idx_a7), ("A7_resid_c", idx_a7c)]:
                    w_sel[(H, p)][name].append(float(np.mean(wH[idx])))
                    accs[(H, p)][name].append(eval_arm_krr(Et, Ee, yt, ye, idx))

                if H == 2 and p == 16:
                    proj_a3.append(one_projection_curve(Et, idx_a3, 16))
                    proj_a7.append(one_projection_curve(Et, idx_a7, 16))
                    proj_var.append(one_projection_curve(Et, idx_var, 16))

    p_a3_mean = np.mean(proj_a3, axis=0).tolist()
    p_a7_mean = np.mean(proj_a7, axis=0).tolist()
    p_var_mean = np.mean(proj_var, axis=0).tolist()
    step90_a3 = next((i + 1 for i, v in enumerate(p_a3_mean) if v >= 0.90), None)
    step90_a7 = next((i + 1 for i, v in enumerate(p_a7_mean) if v >= 0.90), None)

    out = {
        "unit": f"{ds_name}_n{n}_d{d}_e{ent}_bw{bw}",
        "ds": ds_name, "n": n, "d": d, "ent": ent, "bw": bw, "n_splits": n_splits,
        "sklearn_version": sklearn.__version__,
        "dilution": {
            "n_w1": int(np.sum(mask_w1)), "n_w2": int(np.sum(mask_w2)),
            "tot_w1": float(np.mean(tot_w1)), "tot_w2": float(np.mean(tot_w2)),
            "tot_ratio_w2_w1": float(np.mean(tot_w2) / np.mean(tot_w1)),
            "mean_w1": float(np.mean(stats_decomp["w1"]["gamma"])),
            "mean_w2": float(np.mean(stats_decomp["w2"]["gamma"])),
            "mean_ratio_w1_w2": float(np.mean(stats_decomp["w1"]["gamma"]) / np.mean(stats_decomp["w2"]["gamma"])),
        },
        "decomp": {
            k: {
                "gamma": float(np.mean(v["gamma"])),
                "mu2": float(np.mean(v["mu2"])),
                "var": float(np.mean(v["var"])),
                "mu2_share_pct": float(np.mean(v["mu2"]) / np.mean(v["gamma"]) * 100.0),
            }
            for k, v in stats_decomp.items()
        },
        "proj_ones_p16": {
            "A3_topp": [round(x, 4) for x in p_a3_mean],
            "A7_resid": [round(x, 4) for x in p_a7_mean],
            "A3_topvar": [round(x, 4) for x in p_var_mean],
            "step_gt_0p90_A3": step90_a3,
            "step_gt_0p90_A7": step90_a7,
        },
        "results": {},
    }
    for H in (2,):
        for p in (16, 64):
            key_hp = f"H{H}_p{p}"
            out["results"][key_hp] = {
                name: {
                    "acc_mean": float(np.mean(accs[(H, p)][name])),
                    "acc_std": float(np.std(accs[(H, p)][name], ddof=1)),
                    "w_mean": float(np.mean(w_sel[(H, p)][name])),
                }
                for name in ("A3_topp", "A3_topvar", "A7_resid", "A7_resid_c")
            }
            out["results"][key_hp]["d_topvar_topp"] = float(np.mean(np.array(accs[(H, p)]["A3_topvar"]) - np.array(accs[(H, p)]["A3_topp"])))
            out["results"][key_hp]["d_A7_topp"] = float(np.mean(np.array(accs[(H, p)]["A7_resid"]) - np.array(accs[(H, p)]["A3_topp"])))
            out["results"][key_hp]["d_A7_topvar"] = float(np.mean(np.array(accs[(H, p)]["A7_resid"]) - np.array(accs[(H, p)]["A3_topvar"])))
    out["results"]["p16"] = out["results"]["H2_p16"]
    out["results"]["p64"] = out["results"]["H2_p64"]
    return out


def compute_g5_bootstraps():
    rf_dir = HOPE / "runs" / "t11_v3"
    files_8_12 = ["t11v3_FULL_results.json", "t11fmv3_FULL_results.json"]
    files_all = ["t11v3_FULL_results.json", "t11fmv3_FULL_results.json",
                 "t11iobign16v3_FULL_results.json", "t11fmbign16v3_FULL_results.json"]
    out = {}
    for label, flist in [("n8_12", files_8_12), ("n8_16", files_all)]:
        rows, ecs = [], {}
        for rf in flist:
            d = json.loads((rf_dir / rf).read_text(encoding="utf-8"))
            rows.extend(d["rows"])
            ecs.update(d["ec"])
        G1SET = [r for r in rows if r["live"] and r["ec"] == "EC-LEARN" and r["k6_ok"]]
        out[label] = {}
        for arm in ("A3_topp", "A7_resid"):
            r_s1000, keys_s1000 = [], []
            r_sself, keys_sself = [], []
            for r in G1SET:
                mtr, mte = ecs[r["key"]]["m_tr"], ecs[r["key"]]["m_te"]
                M = mtr + mte
                sa, sb = float(r["S_A2_hbody"]), float(r[f"S_{arm}"])
                if np.isfinite(sa) and np.isfinite(sb):
                    r_s1000.append((sa * M) / (sb * M + 1000.0 * mtr))
                    keys_s1000.append(r["key"])
                    r_sself.append((sa * M) / (sb * (M + mtr)))
                    keys_sself.append(r["key"])
                elif np.isinf(sa) and np.isfinite(sb):
                    r_s1000.append(1000.0)
                    keys_s1000.append(r["key"])
                    r_sself.append(1000.0)
                    keys_sself.append(r["key"])
                elif np.isfinite(sa) and np.isinf(sb):
                    r_s1000.append(1.0 / 1000.0)
                    keys_s1000.append(r["key"])
                    r_sself.append(1.0 / 1000.0)
                    keys_sself.append(r["key"])

            def row_and_cluster_boot(vals, keys, n_boot=4000, seed=0):
                vals = np.asarray(vals, dtype=float)
                rng = np.random.default_rng(seed)
                med = float(np.median(vals))
                rb = np.median(vals[rng.integers(0, len(vals), (n_boot, len(vals)))], axis=1)
                r_lo, r_hi = float(np.percentile(rb, 2.5)), float(np.percentile(rb, 97.5))
                ukeys = sorted(set(keys))
                key_to_idx = {u: [i for i, k in enumerate(keys) if k == u] for u in ukeys}
                rng_c = np.random.default_rng(seed)
                cb = []
                for _ in range(n_boot):
                    picked = rng_c.choice(ukeys, size=len(ukeys), replace=True)
                    idxs = [i for u in picked for i in key_to_idx[u]]
                    cb.append(np.median(vals[idxs]))
                c_lo, c_hi = float(np.percentile(cb, 2.5)), float(np.percentile(cb, 97.5))
                return {"n_rows": len(vals), "n_units": len(ukeys), "median": med,
                        "row_ci": [r_lo, r_hi], "unit_ci": [c_lo, c_hi]}

            out[label][arm] = {
                "S_sel_1000_finite_only": row_and_cluster_boot(
                    [x for x, r in zip(r_s1000, G1SET) if np.isfinite(float(r["S_A2_hbody"])) and np.isfinite(float(r[f"S_{arm}"]))],
                    [r["key"] for r in G1SET if np.isfinite(float(r["S_A2_hbody"])) and np.isfinite(float(r[f"S_{arm}"]))],
                ),
                "S_sel_1000_all_g1set": row_and_cluster_boot(r_s1000, keys_s1000),
                "S_sel_self_g5": row_and_cluster_boot(r_sself, keys_sself),
            }
    return out


if __name__ == "__main__":
    t0 = time.time()
    print(f"Running with sklearn={sklearn.__version__}")
    # p=16'da A3 < A2_hbody olan tum n=8,10 birimleri (breast_cancer, ionosphere, fashion_mnist) + kontrol
    target_units = [
        ("breast_cancer", 8, 8, 0.5, 0.15, 30),
        ("breast_cancer", 8, 8, 1.0, 0.15, 30),
        ("breast_cancer", 10, 8, 0.5, 0.15, 15),
        ("ionosphere", 8, 8, 0.5, 0.15, 30),
        ("ionosphere", 10, 8, 0.5, 0.15, 15),
        ("fashion_mnist", 10, 8, 0.5, 0.15, 15),
        ("ionosphere", 8, 8, 1.0, 0.25, 30),
        ("ionosphere", 10, 8, 1.0, 0.15, 15),
    ]
    unit_reports = []
    for ds_name, n, d, ent, bw, n_sp in target_units:
        u_t0 = time.time()
        rep = run_unit_analysis(ds_name, n, d, ent, bw, n_splits=n_sp)
        unit_reports.append(rep)
        print(f"[{rep['unit']}] ({time.time()-u_t0:.1f}s, splits={n_sp})")
        for H in (2,):
            r16 = rep["results"][f"H{H}_p16"]
            r64 = rep["results"][f"H{H}_p64"]
            print(f"  H={H} p=16: A3={r16['A3_topp']['acc_mean']:.2f}% | TopVar={r16['A3_topvar']['acc_mean']:.2f}% | A7={r16['A7_resid']['acc_mean']:.2f}% | A7c={r16['A7_resid_c']['acc_mean']:.2f}%  (d_A7_A3={r16['d_A7_topp']:+.2f}, d_Var_A3={r16['d_topvar_topp']:+.2f}, d_A7_Var={r16['d_A7_topvar']:+.2f})")
            print(f"  H={H} p=64: A3={r64['A3_topp']['acc_mean']:.2f}% | TopVar={r64['A3_topvar']['acc_mean']:.2f}% | A7={r64['A7_resid']['acc_mean']:.2f}% | A7c={r64['A7_resid_c']['acc_mean']:.2f}%  (d_A7_A3={r64['d_A7_topp']:+.2f}, d_Var_A3={r64['d_topvar_topp']:+.2f}, d_A7_Var={r64['d_A7_topvar']:+.2f})")

    g5_boot = compute_g5_bootstraps()
    payload = {
        "created_at": "2026-10-17",
        "sklearn_version": sklearn.__version__,
        "units": unit_reports,
        "g5_bootstrap": g5_boot,
    }
    out_path1 = HOPE / "runs" / "mean_share_real" / "dc_offset_and_topvar.json"
    out_path1.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved to {out_path1} in {time.time()-t0:.1f}s")
