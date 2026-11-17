"""Like-for-like ordered Pauli product comparison; saves separate results."""
import benchmark as base
from scipy.sparse.linalg import expm_multiply
import numpy as np
import csv
import json


def sparse_terms(n, terms):
    result = []
    for term, coefficient in terms:
        label = ['I']*n
        for j, axis in term:
            label[n-1-j] = axis
        result.append((''.join(label), coefficient))
    return base.SparsePauliOp.from_list(result).to_matrix(sparse=True).tocsr()

def evolve(matrix, t, states):
    return expm_multiply(-1j*t*matrix, states, traceA=-1j*t*matrix.diagonal().sum())

def system_product(n, terms, shift, time, order, repetitions):
    qc = base.QuantumCircuit(n)
    for pauli, coeff in base.product_terms(terms, order, repetitions):
        pre, rotation, post = base.pauli_gadget(pauli, 2*time*coeff)
        for op in pre+[rotation]+post:
            base.append_op(qc, op, offset=0)
    qc.global_phase = -time*shift
    return qc


def main():
    rows = []
    old = json.loads((base.OUT/'details.json').read_text(encoding='utf-8'))
    previous = {c['n']: c['paper_reverse_overlap_component_max_difference']
                for c in old['checks'] if 'paper_reverse_overlap_component_max_difference' in c}
    for n in base.SIZES:
        params = np.load(base.OUT/f'parameters_n{n}.npz')
        inputs = [0, (1 << (n//2-1)) | (1 << (n//2))]
        prepared = []
        for key in ('theta', 'theta_m'):
            qc = base.QuantumCircuit(n)
            for op in base.pqc_ops(n, params[key]): base.append_op(qc, op, offset=0)
            prepared.append([base.Statevector.from_int(i, 2**n).evolve(qc) for i in inputs])
        left, right = prepared
        terms, shift = base.hn_terms(n)
        generator = sparse_terms(n, terms+[((),shift)])
        exact = [np.vdot(l.data, evolve(generator, base.DT, r.data)) for l,r in zip(left,right)]
        for order in (1, 2):
            for repetitions in (1, 2, 4):
                s = system_product(n, terms, shift, base.DT, order, repetitions)
                s2 = system_product(n, terms, shift, 2*base.DT, order, repetitions)
                # Qiskit 2.2.3 applies circuit global phase in-place to the
                # starting buffer; explicitly copy states before evolution.
                original = np.array([np.vdot(l.data, r.copy().evolve(s).data) for l,r in zip(left,right)])
                reverse = np.array([np.vdot(l.copy().evolve(s).data, r.copy().evolve(s2).data) for l,r in zip(left,right)])
                delta = reverse-original
                error = float(max(np.max(abs(delta.real)), np.max(abs(delta.imag))))
                if order == 1 and repetitions == 1:
                    assert abs(error-previous[n]) < 1e-12, (n,error,previous[n],original,reverse)
                row = dict(n=n, order=order, repetitions=repetitions,
                    overlap_component_difference=error,
                    overlap_complex_difference=float(np.max(abs(delta))),
                    original_exact_overlap_error=float(np.max(abs(original-exact))),
                    reverse_exact_overlap_error=float(np.max(abs(reverse-exact))))
                # Count and cross-check both Hadamard circuits at one step.
                if repetitions == 1:
                    circuits, _ = base.circuits(n, order, repetitions, save_parameters=False)
                    for variant, target in [('original',original), ('paper_reverse',reverse)]:
                        circuit = circuits[variant]
                        for i, state in enumerate(inputs):
                            measured = base.ancilla_x(circuit,state)+1j*base.ancilla_x(circuit,state,True)
                            assert abs(measured-target[i]) < 1e-12, (n,order,variant,measured,target[i])
                        compiled = base.transpile(circuit, basis_gates=['u','cx'],
                            coupling_map=base.CouplingMap.from_line(n+1,bidirectional=True),
                            initial_layout=list(range(n+1)), routing_method='sabre',
                            seed_transpiler=base.SEED, optimization_level=base.OPT,num_processes=1)
                        circuit_dir = base.OUT / 'circuits'
                        circuit_dir.mkdir(exist_ok=True)
                        base.qasm2.dump(compiled, str(circuit_dir/f'{variant}_order{order}_n{n}.qasm'))
                        counts = base.counts(compiled)
                        for ins in compiled.data:
                            if len(ins.qubits)==2:
                                ids=[compiled.find_bit(q).index for q in ins.qubits]
                                assert abs(ids[0]-ids[1])==1
                        for k in ('single','double','depth'):
                            row[variant+'_'+k] = counts[k]
                rows.append(row)
                print(row,flush=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (base.OUT/'pauli_order_comparison.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (base.OUT/'pauli_order_config.json').write_text(json.dumps({
        'terms': 'same ordered individual Pauli strings as benchmark.py',
        'second_order_chronological': 'P1 half,...,P(L-1) half, PL full, P(L-1) half,...,P1 half',
        'dt': base.DT,'k':base.K,'g':base.G,'interaction':base.INTERACTION,
        'consistent': 'same order and repetition count for S(dt) and S(2dt)',
        'error': 'max of real/imag overlap difference over the same two tested input states',
        'reference': 'sparse matrix exponential of full Hermitian branch generator',
        'gate_counts': 'r=1 only, same linear routing/basis/compiler settings as original benchmark'
    },indent=2),encoding='utf-8')


if __name__=='__main__': main()
