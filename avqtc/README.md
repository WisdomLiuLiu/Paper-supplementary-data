# Minimal reproduction files

Supplementary code and data for Liu, Huan-Yu, et al. "Simulation of Lindbladian dynamics via adaptive variational quantum trajectory compression." arXiv preprint [arXiv:2607.09051](https://arxiv.org/abs/2607.09051) (2026).

Run each script from its example directory, or run `python reproduce.py` from this directory to redraw all eleven figures. Install the Python dependencies with `python -m pip install -r requirements.txt`. Plotting uses NumPy, SciPy, Matplotlib and `scienceplots`; recomputing the reference dynamics additionally uses QuTiP. A LaTeX installation with `dvipng` and `newtx` reproduces the original typography; otherwise, the plots use Matplotlib's mathtext. CuPy is optional for GPU recomputation.

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

## Recomputing the numerical results

The numbered simulation and training scripts are included. Run `python reproduce.py --help` for commands that recompute one example in a new output directory. These calculations can take substantially longer than plotting; progress is printed throughout. Failed training attempts and the other independent training seeds are not distributed.

The original calculations used Python 3.9, NumPy 1.26.4, SciPy 1.13.1 and CuPy 13.6.0. The numerical kernels are unchanged. Different software or hardware may produce floating-point differences. Git attributes preserve file bytes because the saved records validate source and data checksums.
