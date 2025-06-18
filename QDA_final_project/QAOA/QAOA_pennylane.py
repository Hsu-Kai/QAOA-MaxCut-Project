##############################################################################
# This demonstration is based on
# https://pennylane.ai/qml/demos/tutorial_qaoa_maxcut

import pennylane as qml
from pennylane import numpy as np

np.random.seed(42)

n_wires = 4
graph = [(0, 1), (0, 3), (1, 2), (2, 3)]
#n_wires = 6
#graph = [(0, 1), (0, 5), (1, 2), (1, 4), (2, 3), (3, 4), (4, 5)]
#n_wires = 8
#graph = [(0, 1), (0, 7), (1, 2), (1, 6), (2, 3), (2, 5), (3, 4), (4, 5), (5, 6), (6, 7)]


# Mixer Hamiltonian
def U_M(beta):
    for wire in range(n_wires):
        qml.RX(2 * beta, wires=wire)

# Cost Hamiltonian
def U_C(gamma):
    for edge in graph:
        qml.CNOT(wires=edge)
        qml.RZ(gamma, wires=edge[1])
        qml.CNOT(wires=edge)

def bitstring_to_int(bit_string_sample):
    return int(2 ** np.arange(len(bit_string_sample)) @ bit_string_sample[::-1])

# quantum device with n qubits
dev = qml.device("lightning.qubit", wires=n_wires, shots=20)

@qml.qnode(dev)
def circuit(gammas, betas, return_samples=False):
    # initialization
    for wire in range(n_wires):
        qml.Hadamard(wires=wire)
    # 2p parameters 
    for gamma, beta in zip(gammas, betas):
        U_C(gamma)
        U_M(beta)

    if return_samples:
        # sample bitstrings to obtain cuts
        return qml.sample()
    # evaluate the objective function C during optimization
    C = qml.sum(*(qml.Z(w1) @ qml.Z(w2) for w1, w2 in graph))
    return qml.expval(C)


# Maximizing the objective function C
# equivalent to minimizing the negative of the objective function C
def objective(params):
    return -0.5 * (len(graph) - circuit(*params))


# Optimizing parameters γ and β
def qaoa_maxcut(n_layers=1):
    print(f"\np={n_layers:d}")

    # initialize parameters and optimizer
    init_params = 0.01 * np.random.rand(2, n_layers, requires_grad=True)
    opt = qml.AdagradOptimizer(stepsize=0.5)

    # update parameters
    params = init_params.copy()
    steps = 100
    for i in range(steps):
        params = opt.step(objective, params)
        if (i + 1) % 5 == 0:
            print(f"Objective after step {i+1:3d}: {-objective(params): .7f}")

    # sample 100 bitstrings
    bitstrings = circuit(*params, return_samples=True, shots=100)

    sampled_ints = [bitstring_to_int(string) for string in bitstrings]

    counts = np.bincount(np.array(sampled_ints))
    most_freq_bit_string = np.argmax(counts)
    print(f"Optimized parameter vectors:\ngamma: {params[0]}\nbeta:  {params[1]}")
    print(f"Most frequently sampled bit string is: {most_freq_bit_string:04b}")

    return -objective(params), sampled_ints


# perform QAOA with p=1,2,......
int_samples1 = qaoa_maxcut(n_layers=1)[1]
int_samples2 = qaoa_maxcut(n_layers=2)[1]
int_samples3 = qaoa_maxcut(n_layers=3)[1]
int_samples4 = qaoa_maxcut(n_layers=4)[1]
int_samples5 = qaoa_maxcut(n_layers=5)[1]
int_samples6 = qaoa_maxcut(n_layers=6)[1]
int_samples7 = qaoa_maxcut(n_layers=7)[1]
int_samples8 = qaoa_maxcut(n_layers=8)[1]
