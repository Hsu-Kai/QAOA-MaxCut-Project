from sklearn.metrics import confusion_matrix
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
#import umap
import seaborn as sns
import pandas as pd
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import networkx as nx

seed = 0
rng = np.random.default_rng(seed=seed)


edges = [(0, 1), (0, 3), (1, 2), (2, 3)]
#edges = [(0, 1), (0, 5), (1, 2), (1, 4), (2, 3), (3, 4), (4, 5)]
#edges = [(0, 1), (0, 7), (1, 2), (1, 6), (2, 3), (2, 5), (3, 4), (4, 5), (5, 6), (6, 7)]
graph = nx.Graph(edges)
positions = nx.spring_layout(graph, seed=1)


for node, coord in positions.items():
    label = 1 if node % 2 == 1 else 0

labels = {node: 1 if node % 2 == 1 else 0 for node in positions}


# Assign labels based on vertex index
labels = {node: 1 if node % 2 == 1 else 0 for node in positions}

# Create a list of colors based on labels (0 -> red, 1 -> blue)
vertex_colors = ['red' if labels[node] == 1 else 'blue' for node in positions]



X = np.array([positions[node] for node in graph.nodes()])
y = np.array([labels[node] for node in graph.nodes()])

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier



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

pred_array = svm_QK.predict(X)


# Compute confusion matrix
#cm = confusion_matrix(true_labels, predicted_labels)

cm = confusion_matrix(y, pred_array)

# Plot confusion matrix: train data
plt.figure(figsize=(8, 6))
#sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
#            xticklabels=np.unique(true_labels), yticklabels=np.unique(true_labels))
sns.heatmap(cm, annot=True, fmt="d", cmap="crest", cbar=False,
            xticklabels=np.unique(y), yticklabels=np.unique(y))

#plt.title("Confusion Matrix")
#plt.xlabel("Predicted Labels")
#plt.ylabel("True Labels")
plt.show()



