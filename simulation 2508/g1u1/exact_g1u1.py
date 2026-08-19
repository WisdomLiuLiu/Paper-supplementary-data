import numpy as np
from math import pi,e
from qutip import *
from openfermion import *
import matplotlib.pyplot as plt

qubit_number = 10
X,Y,Z,iden = sigmax(), sigmay(), sigmaz(), identity(2)
mapping_str_to_op = {'X':X, 'Y':Y, 'Z':Z, 'I':iden}

def string_to_operator(string):
    str_split = string.split(' ')
    op = [iden for _ in range(qubit_number)]
    for str in str_split:
        op[int(str[1:])] = mapping_str_to_op[str[0]]
    return tensor(op)

def fermionic_hn_model(g,u):
    t_right = e**g
    t_left = e**(-g)
    excite_right = [FermionOperator('{}^ {}'.format(x+1,x), t_right) for x in range(qubit_number-1)]
    excite_left = [FermionOperator('{}^ {}'.format(x,x+1), t_left) for x in range(qubit_number-1)]
    interaction = [FermionOperator('{}^ {} {}^ {}'.format(x,x,x+1,x+1), u) for x in range(qubit_number-1)]
    hamiltonian = sum(excite_right + excite_left + interaction)
    return(hamiltonian)

def tuple_to_string(tuple):
    op_str_list = ['{}{}'.format(op[1],op[0]) for op in tuple]
    return ' '.join(op_str_list)

def dict_to_qobj(ham_dict):
    qobj_list = 0
    for key in ham_dict:
        ham_term = string_to_operator(key)
        coeff = ham_dict[key]
        qobj_list += coeff * ham_term
    return qobj_list

def qubit_hn_model(g,U):
    fermionic_model = fermionic_hn_model(g,U)
    qubit_ham = jordan_wigner(fermionic_model)
    ham_dict = qubit_ham.terms
    if () in ham_dict:
        del ham_dict[()]
    real_ham = {}
    imag_ham = {}
    for key in ham_dict:
        new_key = tuple_to_string(key)
        if ham_dict[key].real != 0:
            real_ham[new_key] = ham_dict[key].real
        if ham_dict[key].imag != 0:
            imag_ham[new_key] = ham_dict[key].imag
    return dict_to_qobj(real_ham), dict_to_qobj(imag_ham)

def simulate_dynamics(unitary,psi0):
    exp_list = [[] for _ in range(qubit_number)]
    state_list = [psi0]
    for qubit_index in range(qubit_number):
        exp_list[qubit_index].append(expect(op_list[qubit_index], psi0))
    for _ in range(len(time_list) - 1):
        psi0 = unitary * psi0
        psi0 = psi0 / psi0.norm()
        state_list.append(psi0)
        for qubit_index in range(qubit_number):
            exp_list[qubit_index].append(expect(op_list[qubit_index], psi0))
    return state_list,exp_list

def draw_smplot(data):
    plt.imshow(data,extent=[min(time_list),max(time_list),0,qubit_number],aspect='auto')
    y_tick = [x+0.5 for x in range(qubit_number)]
    y_tick_label = [qubit_number-1-x for x in range(qubit_number)]
    x_tick = [x for x in range(5)]
    plt.xticks(ticks=x_tick,labels=x_tick)
    plt.yticks(ticks=y_tick,labels=y_tick_label)
    plt.xlabel('T')
    plt.ylabel('qubit')
    plt.title('U={}'.format(U))
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    initial_state = tensor(
        basis(2,0),
        basis(2,0),
        basis(2,0),
        basis(2,0),
        basis(2,1),
        basis(2,1),
        basis(2,0),
        basis(2,0),
        basis(2,0),
        basis(2,0)
    )


    dt = 0.2
    time_list = np.arange(0,4.01,dt)
    str_list = [string_to_operator('Z{}'.format(index)) for index in range(qubit_number)]
    op_list = [0.5 - 0.5 * op for op in str_list]

    g = 1
    U = 1
    ham_real, ham_imag = qubit_hn_model(g,U)

    ham = ham_real + 1j * ham_imag

    unitary = (-1j * dt * ham).expm()

    state_list, exp_list = simulate_dynamics(unitary,initial_state)

    draw_smplot(exp_list)

    # np.savetxt('./g1u1/exp_exact_g1u1.txt',exp_list)