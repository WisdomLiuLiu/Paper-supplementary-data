Noiseless Ising reconstruction for Supplementary Figure 4

Run: python reproduce.py
Dependencies: numpy, scipy, matplotlib.

The saved parameters are existing five-layer, six-qubit HEA results. Each layer applies RX-RZ-RX on each qubit and then nearest-neighbor CZ gates. The first 50 parameter rows correspond to t=0,0.1,...,4.9. No optimization or hardware execution is performed by this script.

CSV columns: time, direct normalized evolution reference, noiseless PQC expectation, absolute difference. The direct references were obtained using QuTiP sesolve; the script independently checks them against matrix-exponential propagation. Small solver-tolerance differences are reported.

Outputs: reproduced_comparison.pdf/png and reproduction_verification.json. The original verification.json records the revision audit, including comparison against the previously archived simulator outputs.

Archived sampled results: archived_sampled_vqa_*.txt preserve the previous simulator output. The corresponding local exp_simulate function uses CPUQVM with 500,000 shots per time point. These files therefore contain sampling fluctuations and are not the exact statevector evaluations in the CSV files.

The results and original saved parameters are supplied for reproducibility. The local optimization script uses K=100, whereas the manuscript reports K=80; the provenance of this training setting cannot be established from saved state-preparation parameters alone. This reconstruction does not rerun LCHS training and is independent of that metadata discrepancy.

Independent validation against the original QuTiP HEA implementation at rows 0, 20, and 49 for all three fields agrees to machine precision. These data are also included in the revised manuscript source package.
