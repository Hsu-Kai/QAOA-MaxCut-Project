import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import networkx as nx


seed = 0
rng = np.random.default_rng(seed=seed)


#### plot QEK model structure
dev = qml.device("default.qubit", wires=5, shots=None)
wires = dev.wires.tolist()

# Define layer and ansatz
def layer(x, params, wires, i0=0, inc=1):
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0, j], wires=[wire])
    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])

def ansatz(x, params, wires):
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))

adjoint_ansatz = qml.adjoint(ansatz)

# Define kernel circuit
@qml.qnode(dev, interface="autograd")
def kernel_circuit(x1, x2, params):
    ansatz(x1, params, wires=wires)
    adjoint_ansatz(x2, params, wires=wires)
    return qml.probs(wires=wires)

# Generate random input shapes
num_layers = 6
x1 = np.random.rand(18)         # Shape can be adjusted
x2 = np.random.rand(18)         # Same shape as x1
params = np.random.rand(num_layers, 2, len(wires))  # [layers, (RY/CRZ), wires]

# Draw the circuit
fig, ax = qml.draw_mpl(kernel_circuit)(x1, x2, params)
plt.show()



#### plot QAOA model structure
n_wires = 4
#graph = [(0, 1), (0, 7), (1, 2), (1, 6), (2, 3),
#         (2, 5), (3, 4), (4, 5), (5, 6), (6, 7)]
graph = [(0, 1), (0, 3), (1, 2), (2, 3)]
#graph = [(0, 1), (0, 5), (1, 2), (1, 4), (2, 3), (3, 4), (4, 5)]


# Cost unitary
def U_C(gamma):
    for edge in graph:
        qml.CNOT(wires=edge)
        qml.RZ(gamma, wires=edge[1])
        qml.CNOT(wires=edge)

# Mixer unitary
def U_M(beta):
    for wire in range(n_wires):
        qml.RX(2 * beta, wires=wire)

# Device
dev = qml.device("default.qubit", wires=n_wires)

# QAOA circuit
@qml.qnode(dev)
def qaoa_circuit(gammas, betas):
    for wire in range(n_wires):
        qml.Hadamard(wires=wire)
    for gamma, beta in zip(gammas, betas):
        U_C(gamma)
        U_M(beta)
    return qml.probs(wires=range(n_wires))

# Parameters for drawing (e.g. p = 2 layers)
p = 1
gammas = np.random.rand(p)
betas = np.random.rand(p)

# Draw the circuit
fig, ax = qml.draw_mpl(qaoa_circuit)(gammas, betas)
plt.show()
