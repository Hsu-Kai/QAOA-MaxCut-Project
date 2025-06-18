import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.quantum_info import Statevector
from qiskit_algorithms.optimizers import COBYLA

# --- Define the Max-Cut graph (customize here) ---
#edges = [(0, 1), (0, 3), (1, 2), (2, 3)]
edges = [(0, 1), (0, 5), (1, 2), (1, 4), (2, 3), (3, 4), (4, 5)]
#edges = [(0, 1), (0, 7), (1, 2), (1, 6), (2, 3), (2, 5), (3, 4), (4, 5), (5, 6), (6, 7)]

n_qubits = max(max(i, j) for i, j in edges) + 1

# --- Max-Cut cost function ---
def maxcut_cost(bitstring, edges):
    return sum(1 for i, j in edges if bitstring[i] != bitstring[j])

# --- QAOA circuit builder for arbitrary p ---
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
        for q in range(n_qubits):
            qc.rx(2 * beta, q)
    return qc

# --- Expectation value for given parameters ---
def qaoa_expectation(params, p, edges):
    gammas = params[:p]
    betas = params[p:]
    qc = create_qaoa_circuit(gammas, betas, p, edges)
    backend = Aer.get_backend("aer_simulator_statevector")
    qc = transpile(qc, backend)
    state = Statevector.from_instruction(qc)
    probs = state.probabilities_dict()
    return -sum(maxcut_cost(b, edges) * p for b, p in probs.items())  # negative for minimization

# --- Track optimization progress ---
objective_history = []

def objective_with_tracking(params):
    value = qaoa_expectation(params, p, edges)
    obj_val = -value
    objective_history.append(obj_val)
    if len(objective_history) % 5 == 0:
        print(f"Objective after step {len(objective_history):3}: {obj_val:.7f}")
    return value

# --- Optimization ---
p = 8  # adjust QAOA depth here
initial_params = np.random.uniform(0, np.pi, 2 * p)
optimizer = COBYLA(maxiter=100)
opt_result = optimizer.minimize(fun=objective_with_tracking, x0=initial_params)

# --- Final circuit and statevector ---
opt_gammas = opt_result.x[:p]
opt_betas = opt_result.x[p:]
final_qc = create_qaoa_circuit(opt_gammas, opt_betas, p, edges)
backend = Aer.get_backend("aer_simulator_statevector")
final_qc = transpile(final_qc, backend)
final_state = Statevector.from_instruction(final_qc)
final_probs = final_state.probabilities_dict()
sorted_probs = dict(sorted(final_probs.items(), key=lambda x: x[1], reverse=True))

# --- Display results ---
print("\nOptimized angles:")
print("Gammas:", opt_gammas)
print("Betas: ", opt_betas)

print("\nTop measured bitstrings:")
for bitstring, prob in list(sorted_probs.items())[:10]:
    bits = bitstring[::-1]  # LSB→MSB conversion
    cut_val = maxcut_cost(bits, edges)
    print(f"{bits}  |  Cut = {cut_val}  |  Prob = {prob:.4f}")

# --- Plot probability distribution ---
plt.figure(figsize=(10, 4))
plt.bar(sorted_probs.keys(), sorted_probs.values())
plt.xticks(rotation=90)
plt.title("QAOA Final Output Bitstring Probabilities")
plt.tight_layout()
plt.show()

# --- Plot optimization history ---
plt.figure(figsize=(6, 4))
plt.plot(objective_history, marker='o')
plt.xlabel("Optimization Step")
plt.ylabel("Max-Cut Objective Value")
plt.title("QAOA Objective vs Optimization Step")
plt.grid(False)
plt.tight_layout()
plt.show()

"""
# --- Run optimization for p = 1 to 8 ---
all_histories = {}
plt.figure(figsize=(10, 6))
for p in range(1, 9):
    objective_history = []
    maxiter = 100
    print(f"\n=== Optimizing for p = {p} ===")
    initial_params = np.random.uniform(0, np.pi, 2 * p)
    optimizer = COBYLA(maxiter=maxiter)
    optimizer.minimize(fun=objective_with_tracking, x0=initial_params)

    all_histories[p] = objective_history
    plt.plot(objective_history, label=f"p={p}")

# --- Final Plot ---
plt.xlabel("Optimization Step")
plt.ylabel("Max-Cut Objective Value")
plt.title("QAOA Objective Value vs Optimization Step (p = 1 to 8)")
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.show()
"""
