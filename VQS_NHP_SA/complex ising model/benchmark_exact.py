from qutip import *
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from qutip_qip.circuit import QubitCircuit
from qutip_qip.operations.gates import gate_sequence_product
from scipy.optimize import minimize
import multiprocessing as mp

X,Y,Z,iden = sigmax(), sigmay(), sigmaz(), identity(2)
mapping_str_to_op = {'X':X, 'Y':Y, 'Z':Z, 'I':iden}

qubit_number = 6
initial_state = tensor([basis(2,0) for x in range(qubit_number)])
dt = 0.1
time_list = np.arange(0,15,dt)

def string_to_operator(string):
    str_split = string.split(' ')
    op = [iden for _ in range(qubit_number)]
    for str in str_split:
        op[int(str[1:])] = mapping_str_to_op[str[0]]
    return tensor(op)

observable = sum([string_to_operator('Z{}'.format(i)) for i in range(qubit_number)])/qubit_number


def ising_model_complex_mag(j,g_real,g_imag):
    str_zz = ['Z{} Z{}'.format(i,i+1) for i in range(qubit_number-1)]
    str_x = ['X{}'.format(i) for i in range(qubit_number)]
    op_zz = [string_to_operator(s) for s in str_zz]
    op_x = [string_to_operator(s) for s in str_x]
    ham = -j * sum(op_zz) - (g_real + 1j*g_imag) * sum(op_x)
    return ham

j = 1
g_real = 2

ham_0 = ising_model_complex_mag(j,g_real,0)
ham_05 = ising_model_complex_mag(j,g_real,0.5)
ham_1 = ising_model_complex_mag(j,g_real,1)
ham_2 = ising_model_complex_mag(j,g_real,2)


result_0 = sesolve(ham_0, initial_state, time_list, e_ops=[observable])
result_05 = sesolve(ham_05, initial_state, time_list, e_ops=[observable])
result_1 = sesolve(ham_1, initial_state, time_list, e_ops=[observable])
result_2 = sesolve(ham_2, initial_state, time_list, e_ops=[observable])

expectation_0 = result_0.expect[0]
expectation_05 = result_05.expect[0]
expectation_1 = result_1.expect[0]
expectation_2 = result_2.expect[0]

np.savetxt('./data_exact/exp_0.txt',expectation_0)
np.savetxt('./data_exact/exp_05.txt',expectation_05)
np.savetxt('./data_exact/exp_1.txt',expectation_1)
np.savetxt('./data_exact/exp_2.txt',expectation_2)

plt.plot(time_list, expectation_0, label='gi=0')
plt.plot(time_list, expectation_05, label='gi=0.5')
plt.plot(time_list, expectation_1, label='gi=1')
plt.plot(time_list, expectation_2, label='gi=2')
plt.xlabel('Time')
plt.ylabel('Expectation value')
plt.legend()
plt.show()