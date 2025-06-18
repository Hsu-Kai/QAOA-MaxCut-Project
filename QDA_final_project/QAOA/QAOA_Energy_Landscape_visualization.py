import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector

# --- Define Max-Cut graph (customize as needed) ---
edges = [(0, 1), (0, 3), (1, 2), (2, 3)]
#edges = [(0, 1), (0, 5), (1, 2), (1, 4), (2, 3), (3, 4), (4, 5)]
#edges = [(0, 1), (0, 7), (1, 2), (1, 6), (2, 3), (2, 5), (3, 4), (4, 5), (5, 6), (6, 7)]
n_qubits = max(max(i, j) for i, j in edges) + 1

# --- Max-Cut cost function ---
def maxcut_cost(bits, edges):
    return sum(1 for i, j in edges if bits[i] != bits[j])

# --- Build QAOA circuit for arbitrary p ---
def create_qaoa_circuit(gammas, betas, p, edges):
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]
        for (i, j) in edges:
            qc.cx(i, j)
            qc.rz(2 * gamma, j)
            qc.cx(i, j)
        for i in range(n_qubits):
            qc.rx(2 * beta, i)
    return qc

# --- Expectation value computation ---
def compute_expectation(gammas, betas, p, edges):
    qc = create_qaoa_circuit(gammas, betas, p, edges)
    backend = Aer.get_backend("aer_simulator_statevector")
    state = Statevector.from_instruction(transpile(qc, backend))
    probs = state.probabilities_dict()
    exp_val = 0
    for bitstring, prob in probs.items():
        bits = list(reversed(bitstring))  # LSB to MSB
        cost = maxcut_cost(bits, edges)
        exp_val += cost * prob
    return exp_val

# --- Set QAOA depth and visualization target layer ---
p = 8                      # QAOA depth (change as needed)
layer_to_sweep = 1         # Layer index to visualize (0-based)
gamma_vals = np.linspace(0, np.pi, 50)
beta_vals = np.linspace(0, np.pi, 50)

# --- Fixed parameters for other layers ---
fixed_gammas = [np.pi / 4] * p
fixed_betas = [np.pi / 4] * p

# --- Compute energy landscape ---
exp_grid = np.zeros((len(gamma_vals), len(beta_vals)))
for i, gamma in enumerate(gamma_vals):
    for j, beta in enumerate(beta_vals):
        gammas = fixed_gammas.copy()
        betas = fixed_betas.copy()
        gammas[layer_to_sweep] = gamma
        betas[layer_to_sweep] = beta
        exp_grid[i, j] = compute_expectation(gammas, betas, p, edges)

# --- Plot energy landscape ---
G, B = np.meshgrid(beta_vals, gamma_vals)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(B, G, exp_grid, cmap='viridis')
ax.set_xlabel("γ (Layer {})".format(layer_to_sweep + 1))
ax.set_ylabel("β (Layer {})".format(layer_to_sweep + 1))
ax.set_zlabel("Expected Max-Cut Value")
ax.set_title(f"QAOA Energy Landscape (p={p}, sweeping layer {layer_to_sweep + 1})")
plt.tight_layout()
plt.show()
