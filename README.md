# Label-free capacity-based selection of Pauli terms

Code and data for the paper

> A. E. Aydın, L. Akgün, A. Daşkın, *Mitigating Exponential Concentration in Quantum Kernels by Label-Free Capacity-Based Selection of Pauli Terms* (2026).

The fidelity kernel of a quantum feature map is a sum of Pauli terms. Under exponential concentration its Gram matrix approaches the identity. We select a small number of Pauli terms by their capacity on unlabeled data (top-capacity selection) or by residual capacity (greedy-residual selection), and compare against H-body selection, random selection and the full kernel.

Every number, figure and table in the paper is reproduced from the result files in `runs/` by the scripts in `figures/`, without rerunning any experiment.

## Layout

| Path | Contents |
|---|---|
| `src/` | Core library: encoding, Pauli expectation values, kernels, diagnosis (`qk_core.py`); selection rules and synthetic labels (`phaseb_core.py`); real-data pipeline (`t11_core.py`); shot model (`gate4_core.py`); closed form of the weight-1 and weight-2 expectation values at depth 1 (`depth1_core.py`, Section 6); check of the statevector simulator against PennyLane `default.qubit` (`pennylane_esdegerlik.py`, needs PennyLane). |
| `notebooks/` | The self-contained notebooks that produced the results (each embeds its own copy of the core code). |
| `scripts/` | Noisy-selection experiment, strict-gold counts and dose analysis, Modal launcher for the n = 20 real-data runs. |
| `runs/` | Result files (JSON/CSV) of every run used in the paper. |
| `figures/` | Scripts that draw the figures and compute the reported numbers, and the JSON files they write. |

## Selection rules: names in the code

The result files use short arm codes:

| Code | Paper |
|---|---|
| `A0` | full kernel |
| `A1_random` | random selection |
| `A2_hbody` | H-body selection (H = 1 or 2, the better on the test set) |
| `A3_topp` | top-capacity selection |
| `A4_merge` | merge variant (not reported; see Section 5.1) |
| `A5_super` | label-aware selector |
| `A6_wnorm` | weight-normalized capacity (experiment 1 only, not reported) |
| `A7_resid` | greedy-residual selection |

`H` in the result rows is the maximal Pauli weight of the pool (`w_max` in the paper), `p` the budget. The synthetic label `qlocal` is excluded from every summary (Section 5.1).

## Experiments and result files

| Paper | Notebook / script | Result |
|---|---|---|
| Section 3 (diagnosis), Table 1, Figure 3 | `notebooks/diagnosis/phase_a2_colab.ipynb`, `phase_a2ext_executed.ipynb` | `runs/phase_a2_v1/`, `runs/phase_a2ext_v1/` (`series_by_n.json`: values per n) |
| Appendix A (encoding with n-independent phase spread) | `notebooks/diagnosis/phase_a2r2_colab.ipynb` | `runs/phase_a2r2_v1/`, `runs/phase_a2_v1/fig1_series_ref.csv` |
| Experiment 1 | `notebooks/synthetic/phase_bv3_FULL.ipynb` | `runs/phase_bv3_full_v1/` |
| Experiment 2 | `notebooks/synthetic/phase_cv3_FULL.ipynb` | `runs/phase_cv3_full_v1/` |
| Experiment 3 | `notebooks/synthetic/phase_c2v3_FULL_fastboot.ipynb` | `runs/phase_c2v3_full_v1/` |
| Experiment 4 (redundancy doses) | `notebooks/synthetic/phase_dv3_FULL.ipynb` | `runs/phase_dv3_full_v1/` |
| Table 4 (two-qubit gates) | `notebooks/shots/gate4_g1rerunv3_FULL.ipynb` | `runs/gate4_g1rerunv3/` |
| Section 5.3 (real data, n = 8–20) | `notebooks/real/t11*_FULL.ipynb`; n = 20: `notebooks/real/d1off_t11*bign20v3_FULL.ipynb` (the notebooks that ran, with the secondary Huang et al. diagnostics switched off), launched on Modal with `scripts/modal_n20/` | `runs/t11_v3/` |
| Section 2.4 (share of the mean on real data) | `scripts/mean_share_real/measure_dc_and_topvar.py` | `runs/mean_share_real/dc_offset_and_topvar.json` |
| Selection inputs measured with finitely many shots | `scripts/noisy_selection/` | `runs/noisy_selection_v1/` (`raw/`: per-unit accuracies, `units/`: unit definitions, `results.json`) |
| Strict-gold counts, label-free vs. label-aware sets, doses | `scripts/selection_bias/` | `runs/selection_bias_check_v1/` |

The synthetic and real-data notebooks ran on GPUs (Google Colab T4/L4/A100, Modal A100) with CuPy; they fall back to NumPy without a GPU but are slow at n ≥ 12. They write their results to Google Drive when run on Colab and to the working directory otherwise.

## Reproducing the figures and numbers

```bash
pip install -r requirements.txt
cd figures
R=..   # repository root

python make_fig_gram.py $R                      # Figure 1
python make_fig_schematic.py $R                 # Figure 2
python make_sreq_slopes.py $R && python make_fig1.py    # Figure 3, Table 1 (shots)
python make_slope_ci.py $R                      # Table 1: bootstrap check (Section 3.1)
python make_mean_share.py $R                    # Section 2.4: share of the mean
python make_fig_gates.py                        # Figure 4
python make_diag_inflation.py $R                # Section 4.3: diagonal inflation
python make_fig_greedy.py                       # Figure 5
python make_pool_fraction.py $R                 # Section 5.1: pool subsampling
python make_fig_candidates.py $R/runs/t11_v3 $R/runs/noisy_selection_v1/results.json $R/runs   # Figures 6, 9, B.1
python make_gate_table.py $R/runs/gate4_g1rerunv3/g1rerunv3_FULL_results.json                  # Table 4
python make_fig_real.py $R/runs/t11_v3 $R/runs/noisy_selection_v1/results.json                 # Figures 7, 8, 10, Table 6
python make_a2r2_sreq.py $R && python make_fig2.py   # Figure A.1
```

Each script writes its figure as PDF/PNG and every plotted value as JSON next to itself; the JSON files in `figures/` are the ones behind the paper. Strict-gold differences and dose statistics:

```bash
cd runs && python ../scripts/selection_bias/syn52.py      # Section 5.2: top-capacity vs. H-body on strict-gold rows
cd .. && python scripts/selection_bias/d23.py               # Section 4.4: doses
```

The noisy-selection scripts load the core library from `$HOPE_ROOT/src`; set `HOPE_ROOT` to the repository root. The interval of the 5.08× saving (Section 5.3) is reproduced by

```bash
HOPE_ROOT=. python scripts/noisy_selection/nsel_analyze.py runs/noisy_selection_v1/raw runs/t11_v3
```

## Notes

- Code comments and some notebook text are in Turkish. Variable names and result fields are in English.
- Shot counts of the full kernel are per kernel entry (overlap test); those of the selected kernels are per input (classical shadows), see Sections 3.1 and 5.
- Large intermediate files (checkpoints, cached kernels) are not included; every reported number is computed from the files in `runs/`, and the n = 20 runs were split between two GPU platforms (Colab and Modal). `scripts/n20_consistency/` checks that this does not matter, for one split (ionosphere, n = 20, ε = 1, bandwidth 0.15) recomputed on a Modal T4 GPU (`runs/n20_consistency/comparison_gpu_t4.json`): the inputs, the selected terms and the full-kernel accuracies agree exactly, and the other accuracies differ by whole test points only, from ties in single precision. The largest effect on the 30-split mean is 0.094 points for the full kernel, H-body, top-capacity, greedy-residual and label-aware selection, 0.19 points for random selection at finite shots, and 0.28 points for the unreported merge variant. A recomputation on CPU in double precision gives the same picture (`comparison_cpu_f64.json`).

## License

MIT (see `LICENSE`). If you use this code or data, please cite the paper.
