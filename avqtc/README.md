# Minimal reproduction files

Supplementary code and data for Liu, Huan-Yu, et al. "Simulation of Lindbladian dynamics via adaptive variational quantum trajectory compression." arXiv preprint [arXiv:2607.09051](https://arxiv.org/abs/2607.09051) (2026).

Run each script from its example directory, or run `python reproduce.py` from this directory to redraw all thirteen figures. Install the Python dependencies with `python -m pip install -r requirements.txt`. Plotting uses NumPy, SciPy, Matplotlib and `scienceplots`; recomputing the reference dynamics additionally uses QuTiP. A LaTeX installation with `dvipng` and `newtx` reproduces the original typography; otherwise, the plots use Matplotlib's mathtext. CuPy is required for the twelve-qubit GPU recomputation and optional for the other examples.

Only training seed `202506`, used in the figures, is included. The XY/XXZ configuration files preserve the original three-seed specification for consistency with the saved data fingerprints; `reproduce.py` selects only `202506`. The 100 trajectory-sampling repetitions used for the means and error bars in figure 01 are all retained.

## XY model

Working directory: `test_xy_r1`

Run `python 08_plot.py`. The five figures are written to `data/figures/figures_seed_202506/`.

- `figure_01_sampling_imbalance.pdf`: sampling means and fluctuations. Required data: `data/reference/reference.npz`.
- `figure_02_b32_training_replacement.pdf`: B32 training depth, HSF and replacement error. Required data:
  - reference curves: `data/reference/reference.npz` and `data/baseline/baseline.npz`
  - selected PQCs: `data/training/b32_direct/seed_202506/manifest.json` and `data/training/b32_iterative/seed_202506/manifest.json`
  - replacement curves: `data/replacement/b32_direct/seed_202506/K20/curves.npz` and `data/replacement/b32_iterative/seed_202506/K{10,15,20}/curves.npz`
- `figure_03_full_training_replacement.pdf`: full-space training and replacement results. Uses the corresponding `full_direct` and `full_iterative` directories and the same reference curves.
- `figure_04_b32_gate_resources.pdf`: single- and two-qubit gate counts. Uses `resources.csv` in the B32 replacement directories listed above.
- `figure_05_full_gate_resources.pdf`: single- and two-qubit gate counts. Uses `resources.csv` in the corresponding full-space replacement directories.

The accompanying reference and baseline metadata, `data/baseline/branches.npz`, and replacement `summary.json` files identify the paired trajectories and validate the saved inputs. The plot script loads all four libraries; their HSF values determine the common fidelity-axis range.

## Disordered XXZ model

Working directory: `test_xxz_r1`

Run `python 08_plot.py`. The five figures are written to `data/figures/figures_seed_202506/`.

- `figure_01_sampling_imbalance.pdf`: sampling means and fluctuations, using `data/reference/reference.npz`.
- `figure_02_b32_training_replacement.pdf`: B32 training and replacement results.
- `figure_03_full_training_replacement.pdf`: full-space training and replacement results.
- `figure_04_b32_gate_resources.pdf`: B32 single- and two-qubit gate counts.
- `figure_05_full_gate_resources.pdf`: full-space single- and two-qubit gate counts.

The required data follow the same directory structure as the XY example. `data/model.json` records the eight-qubit XXZ Hamiltonian and the fixed disorder realization. The library manifests contain the selected PQC parameters and the optimization settings.

## Finite-shot XXZ insertion

Working directory: `test_xxz_full_shots`

- `figure_shots_insert_full_width.pdf`: run `python 07_plot_insert.py`. The output is written to `data/insert/figures/`. Required data:
  - selected depths and post-training HSF: `data/insert/inputs/selected_libraries.json`
  - insertion curves: `data/insert/shots_{infinite,256,1024,4096,16384}/curves.npz`
  - insertion metadata: `data/insert/manifest.json` and the corresponding `summary.json` files
- `data/training/seed_202506/s*/shots_*/checkpoint.json` retains the selected parameters for recomputing insertion. `06_insert_pqc.py` uses these parameters and the paired trajectories in the sibling `test_xxz_r1/data/` directory; specify a new `--output` directory when recomputing insertion.

Shots affect PQC training only. The insertion trajectories are evaluated without execution shot noise or hardware noise. `shots=0` denotes the infinite-shot control. Training-cost analyses and additional diagnostic figures are not included.

## Appendix E.1: Algebraic compression of the XY model

Working directory: `test_xy_algebraic`

Run `python 03_plot.py`, or `python reproduce.py --example algebraic` from the parent directory. The output is `data/figures/figure_xy_algebraic_compression.pdf` (Fig. 16): imbalance error relative to Kraus evolution, mean single-qubit gate count, and mean CNOT count. Shading represents one population standard deviation across the 128 trajectories.

`data/curves.npz` and `data/resources.csv` contain the curves and individual trajectory results; `data/summary.json` records the simulation settings and equivalence checks. The input trajectories and reference dynamics are reused from `../test_xy_r1/data/`, with checksums in `input_checksums.json`.

To recompute, run `python reproduce.py --mode recompute --example algebraic --output /path/to/new_algebraic` from the parent directory. This compiles consecutive no-jump segments using fusion, commutation and turnover identities, evaluates the same 100 times and 128 trajectories per time, and redraws the figure. The compiler specializes the method of Kökcü et al., [arXiv:2303.09538](https://arxiv.org/abs/2303.09538), to isotropic XY gates. The F3C-derived circuit transformations retain their attribution and license in `algebraic.py` and `F3C_LICENSE.txt`.

## Appendix E.2: XXZ system-size dependence

Working directory: `test_xxz_n_scaling`

Run `python 03_plot.py`, or `python reproduce.py --example scaling` from the parent directory. The output is `data/figures/figure_xxz_size_depth_comparison.pdf` (Fig. 17): (a) selected PQC depth versus evolution-block length for `n=6,8,10,12`; (b) best recorded HSF at `s=20` and fixed depths 9 and 10. The untested `n=6`, depth-10 point is left blank. Panel data are exported as CSV files.

The selected parameters and depth-search results are stored in `data/n{06,10}/training/full_direct/seed_202506/manifest.json` and `data/sector_gpu/n12/manifest.json`. The eight-qubit library is reused from `../test_xxz_r1/data/training/full_direct/seed_202506/manifest.json`. Completed attempts at `s=20`, depths 9 and 10 are retained for panel (b); the eight-qubit attempts are in `data/n08/attempts/`.

All four sizes use full-space direct training, seed `202506`, `s=1,...,20`, and L-BFGS-B with at most 100 iterations per initialization (`ftol=1e-12`, `gtol=1e-8`, `maxls=20`). At each depth, all available initializations are optimized before accepting the first depth whose best HSF reaches 0.99. Each new `s` starts at depth 1. The Hamiltonian, disorder generation, initialization and optimizer settings are in `config.json` and the library manifests.

To recompute all four sizes, run `python reproduce.py --mode recompute --example scaling --output /path/to/new_scaling` from the parent directory. Sizes 6, 8 and 10 use the original dense implementation; size 12 uses exact traces and gradients summed over all excitation sectors on GPU. Install the CuPy package matching the local CUDA installation for this step. To run individual sizes, use `python 01_train_dense.py --sizes 6 8 10 --output /path/to/new_scaling` and `python 02_train_sector.py --sizes 12 --output /path/to/new_scaling`. The latter requires a CUDA GPU. Replot using `python 03_plot.py --output /path/to/new_scaling`. Training prints progress and supports resuming completed attempts.

## Recomputing the numerical results

The numbered simulation and training scripts are included. Run `python reproduce.py --help` for commands that recompute one example in a new output directory. These calculations can take substantially longer than plotting; progress is printed throughout. The distribution retains the selected libraries and the completed fixed-depth attempts needed for Fig. 17(b); other attempt histories and training seeds are omitted.

The original calculations used Python 3.9, NumPy 1.26.4, SciPy 1.13.1 and CuPy 13.6.0. The numerical kernels are unchanged. Different software or hardware may produce floating-point differences. Git attributes preserve file bytes because the saved records validate source and data checksums.
