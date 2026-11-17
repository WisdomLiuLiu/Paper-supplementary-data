
import numpy as np
from math import pi,e
from qutip import *
import multiprocessing as mp
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations.gates import gate_sequence_product
from scipy.optimize import minimize
from datetime import datetime
from functools import partial

qubit_number = 10
X,Y,Z,iden = sigmax(), sigmay(), sigmaz(), identity(2)
mapping_str_to_op = {'X':X, 'Y':Y, 'Z':Z, 'I':iden}

initial_state = tensor(
    basis(2,0),basis(2,0),basis(2,0),basis(2,0),
    basis(2,1),basis(2,1),
    basis(2,0),basis(2,0),basis(2,0),basis(2,0)
    )

dt = 0.2
time_list = np.arange(0,4.01,dt)

def string_to_operator(string):
    str_split = string.split(' ')
    op = [iden for _ in range(qubit_number)]
    for str in str_split:
        op[int(str[1:])] = mapping_str_to_op[str[0]]
    return tensor(op)

str_list = [string_to_operator('Z{}'.format(index)) for index in range(qubit_number)]
op_list = [0.5 - 0.5 * op for op in str_list]

total_occ_op = sum(op_list)


data = np.loadtxt('unitary_lchs.txt',dtype=complex)
dims = [[2]*10, [2]*10]
unitary_lchs = Qobj(data, dims=dims)


layer = 5
total_number = qubit_number * layer * 3

gate_template = []
for _ in range(layer):
    # 单比特旋转：Z - X - Z
    for idx in range(qubit_number):
        gate_template.append(('RX', idx))
        gate_template.append(('RZ', idx))
        gate_template.append(('RX', idx))

    # 纠缠层1：0-1, 2-3, 4-5, ...
    for idx in range(0, qubit_number - 1, 2):
        gate_template.append(('CZ', idx, idx + 1))

    # 纠缠层2：1-2, 3-4, 5-6, ...
    for idx in range(1, qubit_number - 1, 2):
        gate_template.append(('CZ', idx, idx + 1))


def build_circuit_with_params(theta):
    qc = QubitCircuit(qubit_number)
    t_idx = 0
    for gate in gate_template:
        if gate[0] in ('RX', 'RZ'):
            qc.add_gate(gate[0], targets=gate[1], arg_value=float(theta[t_idx]))
            t_idx += 1
        elif gate[0] == 'CZ':
            qc.add_gate('CZ', targets=gate[2], controls=gate[1])
    return qc

def run_ansatz_state(theta, init_state):
    qc = build_circuit_with_params(theta)
    return qc.run(init_state)

def cost_function_forgra(theta,psimdt):
    ansatz_state = run_ansatz_state(theta, initial_state)
    inner_val = ansatz_state.dag() * psimdt
    fidelity_val = abs(inner_val)
    total_occ = expect(total_occ_op,ansatz_state)
    add_term = abs(total_occ - 2) ** 2
    return 1 - fidelity_val + 10 * add_term

def cost_function(theta,psimdt):
    ansatz_state = run_ansatz_state(theta, initial_state)
    inner_val = ansatz_state.dag() * psimdt
    fidelity_val = abs(inner_val)
    total_occ = expect(total_occ_op,ansatz_state)
    add_term = abs(total_occ - 2) ** 2
    print(1 - fidelity_val + 10 * add_term,datetime.now())
    return 1 - fidelity_val + 10 * add_term


def gradient_single(theta,index,psimdt):
    theta_p = [x for x in theta]
    theta_m = [x for x in theta]
    delta_x = 0.05
    theta_p[index] += delta_x
    theta_m[index] -= delta_x
    delta_y = cost_function_forgra(theta_p,psimdt) - cost_function_forgra(theta_m,psimdt)
    return delta_y / (2*delta_x)

def gradient(theta, psimdt):
    func = partial(gradient_single, theta, psimdt=psimdt)

    with mp.Pool(processes=150) as pool:
        # 这里 iterable 就是 index 列表
        gradient_list = pool.map(func, range(len(theta)))

    return np.array(gradient_list)


if __name__ == '__main__':


    parameter = [np.zeros(total_number)]
    # parameter = np.loadtxt('param_with.txt').tolist()

    for index in range(len(time_list)-len(parameter)):
        print(len(parameter),'len')
        print(index)
        para = np.random.random((total_number))*pi
        # para = [x for x in parameter[-1]]

        thetapre = parameter[-1]
        psim = run_ansatz_state(thetapre, initial_state)
        psimdt = unitary_lchs * psim
        psimdt = psimdt / psimdt.norm()
        result = minimize(cost_function,para,args=(psimdt,),method='BFGS' ,jac=gradient)
        print(result.fun,result.nit)
        if result.fun < 1e-2:

            parameter.append(list(result.x))

            np.savetxt('param_with.txt',parameter)
