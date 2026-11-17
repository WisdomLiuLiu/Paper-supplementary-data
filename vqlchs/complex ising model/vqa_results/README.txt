Noiseless Ising reconstruction from saved VQA parameters

Run: python reproduce.py
Dependencies: numpy, scipy, matplotlib.

The saved parameters are existing five-layer, six-qubit HEA results. Each layer applies RX-RZ-RX on each qubit and then nearest-neighbor CZ gates. The first 50 parameter rows correspond to t=0,0.1,...,4.9. No optimization or hardware execution is performed by this script.

CSV columns: time, direct normalized evolution reference, noiseless PQC expectation, absolute difference. The direct references were obtained using QuTiP sesolve; the script independently checks them against matrix-exponential propagation. Small solver-tolerance differences are reported.

Outputs: reproduced_comparison.pdf/png and reproduction_verification.json. The original verification.json records the revision audit, including comparison against the previously archived simulator outputs.

Archived sampled results: archived_sampled_vqa_*.txt preserve the previous simulator output. The corresponding local exp_simulate function uses CPUQVM with 500,000 shots per time point. These files therefore contain sampling fluctuations and are not the exact statevector evaluations in the CSV files.

The results and original saved parameters are supplied for reproducibility. The local optimization scripts use K=100, delta k=1, and delta t=0.1. The authors confirmed these settings and corrected K=80 in the manuscript to K=100. The statevector reconstruction does not rerun LCHS training. Newly generated LCHS matrices for both K=100 and K=80 are included, with K=80 provided as a comparison.

Independent validation against the original QuTiP HEA implementation at rows 0, 20, and 49 for all three fields agrees to machine precision.
