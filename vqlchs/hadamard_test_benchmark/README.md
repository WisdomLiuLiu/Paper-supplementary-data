# Hadamard-test simplification benchmark

Numerical gate-count and overlap comparisons for the interacting open-boundary Hatano--Nelson model. The final figure is `figures/hn_second_order_scalability.pdf` (also PNG and SVG). It uses **second-order, one-step** Trotter data only.

## Reproduce

Use Python 3.9--3.12 and install the pinned dependencies:

```bash
python -m pip install -r requirements.txt
python benchmark.py
python pauli_order_comparison.py
python plot_second_order.py
```

The archived CSVs and parameters are already supplied: to redraw the figure, only the last command is needed. Plotting follows the manuscript's SciencePlots style and requires LaTeX with `newtxtext` and `newtxmath`, plus `dvipng`. All scripts resolve files relative to their own directory. No hardware account, job submission, or network connection is required after installing dependencies.

## Setup

- System qubits: 2, 4, 6, 8, 10, plus one interference ancilla.
- HN parameters: right hopping exp(g), left hopping exp(-g), g=1, nearest-neighbor interaction U=1, open boundaries.
- One LCHS branch k=1, time step 0.2. The scalar shift of V is retained as an ancilla phase.
- Two five-layer PQCs: RX-RZ-RX on every qubit, then nearest-neighbor CZ gates in even/odd sublayers. Independent uniform angles in [-pi,pi] use seed `20261117+n`; the saved arrays are `parameters_n*.npz`.
- Each bond's generator terms are ordered XX, YY, XY, YX, ZZ, followed by combined site Z terms. The second-order sequence applies all but the last term for a half step, the last term for a full step, and the preceding terms in reverse order for a half step.
- Qiskit 2.2.3, basis `u,cx`, optimization level 1, SABRE routing, seed 20261117. The physical layout is initially ancilla--q0--...--q(n-1) on a bidirectional line. Gate counts include routing and decomposition into the basis gates, and the initial/final ancilla H gates.

Counts describe **one real-overlap circuit**, not all LCHS branches, shots, penalties, gradients, or optimization iterations. They are compiled gate counts, not device-specific pulse counts or proven optimal counts. The original baseline controls each elementary gate before applying the same general compiler settings.

## Data and circuit variants

`benchmark.py` validates first-order circuits and saves `gate_counts.csv` and `details.json`. Its variants are:

- `original`: elementary controlled gates without manual shared-structure simplification.
- `pqc_only`: simplify only the pair of PQCs.
- `simplified_exact`: shared PQCs plus standard controlled Pauli rotations; fixed conjugating gates do not carry the ancilla control. This is exactly equivalent to the original product-formula circuit.
- `paper_reverse`: shared PQCs plus the manuscript's reverse construction applied to the whole propagation block.

`pauli_order_comparison.py` compares first/second order and 1/2/4 repetitions. It writes `pauli_order_comparison.csv`, its configuration JSON, and compiled one-step circuits in `circuits/`. At fixed order and repetition count, both branches use the same ordered formula. The original overlap uses S(dt); the reverse construction uses S(dt)^dagger S(2dt). At finite product-formula order these may differ. The CSV also compares each overlap against the full Hermitian generator's matrix exponential.

Two input basis states are checked: all zeros and two occupied central neighboring sites. Both are passed through the saved PQCs. The plotted error is the maximum **complex absolute overlap difference** over these two cases (`overlap_complex_difference`), not the largest separate real/imaginary component (`overlap_component_difference`), state infidelity, or a worst-case operator norm.

## Final figure values (second order, one step)

| System qubits | Single gates before | Single gates after | Two-qubit gates before | Two-qubit gates after | Complex overlap deviation |
|---:|---:|---:|---:|---:|---:|
| 2 | 484 | 128 | 568 | 144 | 0.0198899162 |
| 4 | 1308 | 302 | 1683 | 373 | 0.0142869150 |
| 6 | 2132 | 488 | 3153 | 647 | 0.0068112712 |
| 8 | 2954 | 675 | 4755 | 943 | 0.0062193800 |
| 10 | 3778 | 864 | 6357 | 1323 | 0.0025540745 |

The figure's `After` curves use `paper_reverse`. The results show gate-count reductions with small tested overlap deviations across these sizes; the decreasing deviations for these fixed parameter samples do not establish a general decreasing-error scaling law. Additional repetition data are available in the CSV but are not plotted in the final figure.

`figures/plotted_data.csv` contains the plotted values, and `figures/caption.tex` contains the corresponding manuscript caption.
