from pennylane import qaoa
from pennylane import numpy as np
from matplotlib import pyplot as plt
import networkx as nx

#edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
#edges = [(0, 1), (0, 3), (1, 2), (2, 3)]
#edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
edges = [(0, 1), (0, 5), (1, 2), (1, 4), (2, 3), (3, 4), (4, 5)]
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
vertex_colors = ['blue' if labels[node] == 1 else 'red' for node in positions]

# Draw the graph with colored vertices
nx.draw(graph, with_labels=True, pos=positions, node_color=vertex_colors, font_weight='bold', node_size=500)
plt.show()
