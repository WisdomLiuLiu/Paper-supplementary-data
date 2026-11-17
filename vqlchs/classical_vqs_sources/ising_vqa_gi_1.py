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
time_list = np.arange(0,5,dt)


def string_to_operator(string):
    str_split = string.split(' ')
    op = [iden for _ in range(qubit_number)]
    for str in str_split:
        op[int(str[1:])] = mapping_str_to_op[str[0]]
    return tensor(op)

observable = sum([string_to_operator('Z{}'.format(i)) for i in range(qubit_number)])/qubit_number

def imising_lchs(j,g0,g1):
    idenn = tensor([iden for _ in range(qubit_number)])
    str_zz = ['Z{} Z{}'.format(i,i+1) for i in range(qubit_number-1)]
    str_x = ['X{}'.format(i) for i in range(qubit_number)]
    op_zz = [string_to_operator(s) for s in str_zz]
    op_x = [string_to_operator(s) for s in str_x]
    ham_0 = -j * sum(op_zz) - g0 * sum(op_x)
    ham_1 = - g1 * sum(op_x) - g1 * qubit_number * idenn
    return ham_0,ham_1

j,g0,g1 = 1,2,1
ham0,ham1 = imising_lchs(j,g0,g1)

def lchs(k_range,dk):
    k_list = np.arange(-k_range,k_range+dk,dk)
    coeff = [dk / (pi * (1 + k ** 2)) for k in k_list]
    uk = [ (-1j *( ham0 - k * ham1 ) * dt).expm() for k in k_list]
    return sum(coeff[k] * uk[k] for k in range(len(k_list)))

def general_evolve(ope):
    initial_state = tensor([basis(2,0) for x in range(qubit_number)])
    exp_list = [expect(observable,initial_state)]
    for index in range(len(time_list)-1):
        final = ope * initial_state
        initial_state = final / final.norm()
        exp_list.append(expect(observable,initial_state))
    return exp_list

k_range = 100
dk = 1
u_lchs = lchs(k_range,dk)

layer = 5
total_number = qubit_number * layer * 3

def hardware_efficient_ansatz(theta):
    qc = QubitCircuit(qubit_number)
    parameter_index = 0
    for _ in range(layer):
        for index_qubit in range(qubit_number):
            qc.add_gate('RX',targets=index_qubit,arg_value=theta[parameter_index])
            parameter_index += 1
            qc.add_gate('RZ',targets=index_qubit,arg_value=theta[parameter_index])
            parameter_index += 1
            qc.add_gate('RX',targets=index_qubit,arg_value=theta[parameter_index])
            parameter_index += 1
        for index_qubit in range(qubit_number-1):
            qc.add_gate('CZ',targets=index_qubit,controls=index_qubit+1)
    return gate_sequence_product(qc.propagators())

def cost_function(theta,psimdt):
    u_ansatz = hardware_efficient_ansatz(theta)
    ansatz = u_ansatz * initial_state

    fide = fidelity(ansatz,psimdt)
    return 1 - fide

def gradient_single(theta,index,psimdt):
    theta_p = [x for x in theta]
    theta_m = [x for x in theta]
    delta_x = 0.05
    theta_p[index] += delta_x
    theta_m[index] -= delta_x
    delta_y = cost_function(theta_p,psimdt) - cost_function(theta_m,psimdt)
    return delta_y / (2*delta_x)

def gradient(theta,psimdt):
    p = mp.Pool(processes=24)
    p_list = []
    for index in range(len(theta)):
        p_list.append(p.apply_async(gradient_single,args=(theta,index,psimdt)))
    p.close()
    p.join()
    gradient_list = []
    for p in p_list:
        gradient_list.append(p.get())
    return np.array(gradient_list)

if __name__ == '__main__':

    # parameter = [np.zeros((total_number))]
    # for index in range(len(time_list)):
    #     print(index)
    #     para = np.random.random((total_number))*pi
    #     thetapre = parameter[-1]
    #     u_psim = hardware_efficient_ansatz(thetapre)
    #     psim = u_psim * initial_state
    #     psimdt = u_lchs * psim
    #     psimdt = psimdt / psimdt.norm()
    #     result = minimize(cost_function,para,args=(psimdt,),method='BFGS' ,jac=gradient)
    #     print(result.fun,result.nit)
    #     parameter.append(list(result.x))

    # np.savetxt('param_1_layer_6.txt',parameter)


    parameter = np.loadtxt('param_1_layer_5.txt')
    expectation_vqs = []
    for index in range(50):
        u_ansatz = hardware_efficient_ansatz(parameter[index])
        ansatz = u_ansatz*initial_state
        expectation_vqs.append(expect(observable,ansatz))

    # np.savetxt('./data_vqa/expectation_gi_05.txt',expectation_vqs)

    exact = np.loadtxt('./data_lchs/exp_1.txt')
    expectation_exact = [exact[x] for x in range(len(time_list))]
    plt.plot(time_list,expectation_exact,label='g')
    plt.plot(time_list,expectation_vqs,label='VQS')
    plt.xlabel('Time')
    plt.ylabel('Loschmide Amplitude')
    plt.legend()
    plt.show()
