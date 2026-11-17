# Classical VQS source files

These numerical source files contain the original local model, PQC, loss, finite-difference gradient, and SciPy BFGS implementations. They perform classical simulation only. For reproducing the saved results without retraining, use:

```bash
python "../complex ising model/vqa_results/reproduce.py"
python "../simulation 2508/reproduce_vqs.py"
```

The commands above are relative to this directory; the reconstruction scripts themselves resolve their data paths and work from any current directory.

## Ising

`ising_vqa_gi_05.py`, `ising_vqa_gi_1.py`, and `ising_vqa_gi_2.py` retain the source for the three fields. They use six qubits, five PQC layers, dt=0.1, K=100 and delta k=1. Their archived training loops are commented out; the active main blocks load saved parameters and plot comparisons using historical relative paths. Use the portable reconstruction command above for the supplied results. Training source dependencies are NumPy, SciPy, Matplotlib, QuTiP, and qutip-qip.

## HN

`hn_vqa_switch.py` contains the reversed-hopping, eight-particle training routine, loading `unitary_lchs_switch.txt` and saving `param_switch.txt`. The saved parameters and matrix are supplied under `simulation 2508/LR_switch`.

`hn_vqa_with_penalty.py` contains the two-particle penalty-training routine, loading `unitary_lchs.txt` and saving `param_with.txt`. Its matrix corresponds to the forward HN matrix supplied as `simulation 2508/g1u1/unitary_lchs_g1u1.txt`; the saved comparison parameters and observables are under `simulation 2508/supp_0203`.

These HN source files use ten qubits, five PQC layers, dt=0.2, and a normalized finite-step LCHS target. Their implemented objective is `1 - abs(overlap) + 10*(total_occupation-target)^2`. After BFGS returns, parameters are saved only if the total objective is below 0.01. The sources retain their original random initialization and multiprocessing settings (150 workers); adjust resource settings before a new training run. They are supplied as source records; the reconstruction scripts do not retrain or overwrite archived data.
