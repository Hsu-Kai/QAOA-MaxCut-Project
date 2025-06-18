##############################################################################
# This demonstration is based on
# https://pennylane.ai/qml/demos/tutorial_kernels_module

from pennylane import qaoa
from pennylane import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pennylane as qml
import pandas as pd
from sklearn.svm import SVC
import networkx as nx


seed = 0
rng = np.random.default_rng(seed=seed)


edges = [(0, 1), (0, 3), (1, 2), (2, 3)]
#edges = [(0, 1), (0, 5), (1, 2), (1, 4), (2, 3), (3, 4), (4, 5)]
#edges = [(0, 1), (0, 7), (1, 2), (1, 6), (2, 3), (2, 5), (3, 4), (4, 5), (5, 6), (6, 7)]
graph = nx.Graph(edges)
positions = nx.spring_layout(graph, seed=1)

nx.draw(graph, with_labels=True, pos=positions)
plt.show()

for node, coord in positions.items():
    print(f"Node {node}: x = {coord[0]:.4f}, y = {coord[1]:.4f}")

print(positions)

for node, coord in positions.items():
    label = 1 if node % 2 == 1 else 0
    print(f"Node {node}: x = {coord[0]:.4f}, y = {coord[1]:.4f}, label = {label}")

labels = {node: 1 if node % 2 == 1 else 0 for node in positions}
print(labels)


# Assign labels based on vertex index
labels = {node: 1 if node % 2 == 1 else 0 for node in positions}

# Create a list of colors based on labels (0 -> red, 1 -> blue)
vertex_colors = ['red' if labels[node] == 1 else 'blue' for node in positions]

# Draw the graph with colored vertices
nx.draw(graph, with_labels=True, pos=positions, node_color=vertex_colors, font_weight='bold', node_size=500)
plt.show()

X = np.array([positions[node] for node in graph.nodes()])
Y = np.array([labels[node] for node in graph.nodes()])

num_train = 2
num_test = 2
train_indices = rng.choice(len(Y), num_train, replace=False)
test_indices = rng.choice(
    np.setdiff1d(range(len(Y)), train_indices), num_test, replace=False
)

Y_train, Y_test = Y[train_indices], Y[test_indices]
X_train, X_test = X[train_indices], X[test_indices]


# Defining a Quantum Embedding Kernel

def layer(x, params, wires, i0=0, inc=1):
    """Building block of the embedding ansatz"""
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0, j], wires=[wire])

    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])


# To construct the ansatz, this layer is repeated multiple times, reusing
# the datapoint with different variational  parameters 
# the datapoints here are cartesian coordinates of each node with a given graph 

def ansatz(x, params, wires):
    """The embedding ansatz"""
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))


adjoint_ansatz = qml.adjoint(ansatz)


def random_params(num_wires, num_layers):
    """Generate random variational parameters in the shape for the ansatz."""
    return np.random.uniform(0, 2 * np.pi, (num_layers, 2, num_wires), requires_grad=True)


dev = qml.device("default.qubit", wires=5, shots=None)
wires = dev.wires.tolist()


# computing the overlap of the quantum states 

@qml.qnode(dev, interface="autograd")
def kernel_circuit(x1, x2, params):
    ansatz(x1, params, wires=wires)
    adjoint_ansatz(x2, params, wires=wires)
    return qml.probs(wires=wires)

def kernel(x1, x2, params):
    return kernel_circuit(x1, x2, params)[0]

init_params = random_params(num_wires=5, num_layers=6)


#kernel value between different datapoints:

kernel_value = kernel(X[0], X[1], init_params)
print(f"The kernel value between the first and second datapoint is {kernel_value:.3f}")

init_kernel = lambda x1, x2: kernel(x1, x2, init_params)
K_init = qml.kernels.square_kernel_matrix(X, init_kernel, assume_normalized_kernel=True)

with np.printoptions(precision=3, suppress=True):
    print(K_init)




# SVM prediction via sklearn.svm.SVC

svm = SVC(kernel=lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, init_kernel)).fit(X, Y)

def accuracy(classifier, X, Y_target):
    return 1 - np.count_nonzero(classifier.predict(X) - Y_target) / len(Y_target)

accuracy_init = accuracy(svm, X, Y)
print(f"The accuracy of the kernel with random parameters is {accuracy_init:.3f}")




# kernel-target alignment
kta_init = qml.kernels.target_alignment(X, Y, init_kernel, assume_normalized_kernel=True)

print(f"The kernel-target alignment for our dataset and random parameters is {kta_init:.3f}")

def target_alignment(
    X,
    Y,
    kernel,
    assume_normalized_kernel=False,
    rescale_class_labels=True,
):
    """Kernel-target alignment between kernel and labels."""

    K = qml.kernels.square_kernel_matrix(
        X,
        kernel,
        assume_normalized_kernel=assume_normalized_kernel,
    )

    if rescale_class_labels:
        nplus = np.count_nonzero(np.array(Y) == 1)
        nminus = len(Y) - nplus
        _Y = np.array([y / nplus if y == 1 else y / nminus for y in Y])
    else:
        _Y = np.array(Y)

    T = np.outer(_Y, _Y)
    inner_product = np.sum(K * T)
    norm = np.sqrt(np.sum(K * K) * np.sum(T * T))
    inner_product = inner_product / norm

    return inner_product


params = init_params
opt = qml.GradientDescentOptimizer(0.2)

for i in range(500):
    # Choose subset of datapoints to compute the KTA on.
    subset = np.random.choice(list(range(len(X_train))), 1)
    # Define the cost function for optimization
    cost = lambda _params: -target_alignment(
        X_train[subset],
        Y_train[subset],
        lambda x1, x2: kernel(x1, x2, _params),
        assume_normalized_kernel=True,
    )
    # Optimization step
    params = opt.step(cost, params)

    # Report the alignment on the full dataset every 50 steps.
    if (i + 1) % 50 == 0:
        current_alignment = target_alignment(
            X,
            Y,
            lambda x1, x2: kernel(x1, x2, params),
            assume_normalized_kernel=True,
        )
        print(f"Step {i+1} - Alignment = {current_alignment:.3f}")


# prediction using SVC with a trained kernel

trained_kernel = lambda x1, x2: kernel(x1, x2, params)
trained_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel)


svm_train = SVC(kernel=trained_kernel_matrix).fit(X_train, Y_train)
#svm_test = SVC(kernel=trained_kernel_matrix).fit(X_test, Y_test)
pred_array_train = svm_train.predict(X_train)
pred_array_test = svm_train.predict(X_test)
accuracy_train = 1 - np.count_nonzero(pred_array_train - Y_train) / len(Y_train)
accuracy_test = 1 - np.count_nonzero(pred_array_test - Y_test) / len(Y_test)

print(f"The accuracy of a kernel for training sets with trained parameters is {accuracy_train:.3f}")
print(f"The accuracy of a kernel for testing sets with trained parameters is {accuracy_test:.3f}")
np.save("/home/hsukaicheng/Special_Topics_on_Quantum_Design_Automation/final_project/final_weights_v00", params)
np.save("/home/hsukaicheng/Special_Topics_on_Quantum_Design_Automation/final_project/pred_array_train_v00", pred_array_train)
np.save("/home/hsukaicheng/Special_Topics_on_Quantum_Design_Automation/final_project/pred_array_test_v00", pred_array_test)
np.save("/home/hsukaicheng/Special_Topics_on_Quantum_Design_Automation/final_project/train_indices_v00", train_indices)
np.save("/home/hsukaicheng/Special_Topics_on_Quantum_Design_Automation/final_project/test_indices_v00", test_indices)


