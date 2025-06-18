import pennylane as qml
from pennylane import qaoa
from pennylane import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

#edges = [(0, 1), (0, 3), (1, 2), (2, 3)]
edges = [(0, 1), (0, 5), (1, 2), (1, 4), (2, 3), (3, 4), (4, 5)]
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
y = np.array([labels[node] for node in graph.nodes()])

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

clf = SVC(kernel='rbf')  # or try KNeighborsClassifier(), RandomForestClassifier()
clf.fit(X, y)


def plot_decision_boundary_with_edges(X, y, model, ax, title, graph, positions):
    # Mesh grid for decision boundary
    h = 0.01
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Predict over mesh
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot decision regions
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')

    # Plot data points
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k', s=200)

    # Plot graph edges over the decision regions
    nx.draw_networkx_edges(graph, pos=positions, ax=ax, edge_color='gray', width=2)

    # Optional: replot nodes for better visibility
    nx.draw_networkx_nodes(graph, pos=positions, ax=ax, node_color=['blue' if labels[n] else 'red' for n in graph.nodes()], edgecolors='k', node_size=500)

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return scatter



fig, ax = plt.subplots(figsize=(8, 6))

# data
X = np.array([positions[n] for n in graph.nodes()])
y = np.array([labels[n] for n in graph.nodes()])


##############################################################################################################
dev = qml.device("default.qubit", wires=5, shots=None)
wires = dev.wires.tolist()
def layer(x, params, wires, i0=0, inc=1):
    #Building block of the embedding ansatz#
    i = i0
    for j, wire in enumerate(wires):
        qml.Hadamard(wires=[wire])
        qml.RZ(x[i % len(x)], wires=[wire])
        i += inc
        qml.RY(params[0, j], wires=[wire])

    qml.broadcast(unitary=qml.CRZ, pattern="ring", wires=wires, parameters=params[1])

    
def ansatz(x, params, wires):
    #The embedding ansatz#
    for j, layer_params in enumerate(params):
        layer(x, layer_params, wires, i0=j * len(wires))

        
adjoint_ansatz = qml.adjoint(ansatz)

@qml.qnode(dev, interface="autograd")
def kernel_circuit(x1, x2, params):
    ansatz(x1, params, wires=wires)
    adjoint_ansatz(x2, params, wires=wires)
    return qml.probs(wires=wires)

def kernel(x1, x2, params):
    return kernel_circuit(x1, x2, params)[0]

params = np.load('/home/hsukaicheng/Special_Topics_on_Quantum_Design_Automation/final_project/final_weights_v00.npy')

trained_kernel = lambda x1, x2: kernel(x1, x2, params)

trained_kernel_matrix = lambda X1, X2: qml.kernels.kernel_matrix(X1, X2, trained_kernel)
svm_QK = SVC(kernel=trained_kernel_matrix).fit(X, y)

QKpred_array = svm_QK.predict(X)
QKaccuracy = 1 - np.count_nonzero(QKpred_array - y) / len(y)
##############################################################################################################


# Classifier
from sklearn.svm import SVC
#clf = SVC(kernel='rbf')
clf = SVC(kernel=trained_kernel_matrix)
clf.fit(X, y)

# Plot with edges
plot_decision_boundary_with_edges(X, y, clf, ax, "Decision Boundary with Graph Edges", graph, positions)
plt.show()
