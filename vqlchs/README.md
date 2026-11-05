# Minimal reproduction files

Run each notebook from its own directory so that the relative data paths resolve correctly. The plotting notebooks require Python with NumPy, Matplotlib, QuTiP, SciPy, `scienceplots`, and a LaTeX installation. The HN numerical test scripts additionally require OpenFermion.

## SSH model

Working directory: `SSH model  single qubit`

- `ssh_bench_lchs.pdf`: run the first two code cells of `data_plot.ipynb`. Required data:
  - `data_lchs_benchmark_k_dk/40_1.txt`
  - `data_lchs_benchmark_k_dk/40_2.txt`
  - `data_lchs_benchmark_k_dk/80_1.txt`
  - `data_lchs_benchmark_k_dk/80_2.txt`
  - `data_lchs_benchmark_k_dk/exp_exact.txt`
- `ssh_optimize.pdf`: run all cells of `data_plot.ipynb`. Required data:
  - `dt.txt`
  - `2dt.txt`

## Dissipative Ising model

Working directory: `complex ising model`

- `ising_errorbar.pdf`: run all cells of `data_plot.ipynb`. Required data:
  - theoretical curves: `data_exact/exp_05.txt`, `data_exact/exp_1.txt`, and `data_exact/exp_2.txt`
  - repeated measurements: `exp_05/`, `exp_1/`, and `exp_2/`
  - each repeated-measurement directory contains files `0.txt` through `49.txt`, one file per time point; each file contains the repeated results used to calculate the plotted mean and standard deviation.
- `benchmark_exact.py` is the retained minimal numerical model test. It recalculates the exact Ising dynamics and writes the theoretical data into `data_exact/`.

## Interacting Hatano--Nelson model

Working directory: `simulation 2508`

- `hn_num_switch.pdf`: run the first three cells of `data_plot_switch.ipynb`. Required data:
  - `g1u1/exp_exact_g1u1.txt`
  - `g1u1/exp_vqa_g1u1.txt`
  - `LR_switch/exp_exact_switch.txt`
  - `LR_switch/exp_vqa_switch.txt`
- `hn_2026.pdf`: run all cells of `data_plot_switch.ipynb`. In addition to the four files above, it requires:
  - `g1u1/exp_8_2026/` and `g1u1/exp_9_2026/`
  - `LR_switch/exp_8_2026/` and `LR_switch/exp_9_2026/`
  - each directory contains files `0.txt` through `20.txt`, one file per time point; the repeated results in each file are used to calculate the mean and standard deviation.
- `hn_penalty_compare.pdf`: run all cells of `data_plot.ipynb`. Required data:
  - `supp_0203/fidelity_with.txt`
  - `supp_0203/fidelity_without.txt`
  - `supp_0203/total_occ_with.txt`
  - `supp_0203/total_occ_without.txt`
- `g1u1/exact_g1u1.py` and `LR_switch/exact_switch.py` are the retained minimal numerical model tests for the two hopping directions.

## Additional Ising VQA results and LCHS construction

`complex ising model/vqa_results/` contains the stored five-layer PQC parameters, reconstructed noiseless spin-polarization CSVs, archived finite-shot simulator outputs, and `reproduce.py`. Run that script from any working directory to reproduce the additional comparison figure. See its README for provenance and the distinction between sampled and exact circuit expectations.

`build_lchs.py` constructs the finite-step Cauchy-quadrature sum from Hermitian matrix exponentials. It requires NumPy and SciPy. Examples (run from `vqlchs`):

```bash
python build_lchs.py --model ising --gi 0.5 --K 100 --output ising_lchs.txt
python build_lchs.py --model hn --sector 2 --verify "simulation 2508/g1u1/unitary_lchs_g1u1.txt"
python build_lchs.py --model hn --reverse --sector 8 --verify "simulation 2508/LR_switch/unitary_lchs_switch.txt"
```

The two HN TXT matrices are the original saved 1024-by-1024 complex matrices, copied without numerical changes (K=100, delta k=0.5, delta t=0.2). The validation JSON records file hashes and independent fixed-number-block checks. Their historical filenames say `unitary_lchs`, but the weighted sum itself is non-unitary. Without `--sector`, the builder constructs the full HN matrix; this is substantially more expensive than checking a fixed-number block.

The Ising `lchs_matrices/` directory contains newly reconstructed 64-by-64 matrices for gi=0.5,1,2 and both K=80 and K=100, with delta k=1 and delta t=0.1. The local training script specifies K=100 whereas the manuscript reports K=80, so both settings are labeled explicitly. These newly generated matrices are not represented as previously archived files or proof of which setting produced the saved PQC parameters. Scalar Hamiltonian shifts and basis conventions are documented in the builder.
