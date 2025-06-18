
# QAOA MaxCut Project

This project implements and analyzes the **Quantum Approximate Optimization Algorithm (QAOA)** for solving the **MaxCut** problem on unweighted graphs, using both **PennyLane** and **Qiskit** quantum programming frameworks.

---

## 📌 Project Summary

The MaxCut problem is a classical NP-hard optimization problem that involves partitioning the vertices of a graph to maximize the number of edges crossing the partition. QAOA provides a variational quantum-classical hybrid approach for approximating such solutions.

In this project:
- QAOA circuits were implemented using **PennyLane** and reproduced using **Qiskit** and **OpenQASM**.
- Graphs of sizes 4, 6, and 8 nodes were tested.
- Performance was evaluated under varying circuit depths \( p \) to study convergence behavior and approximation quality.
- Circuit structures, energy landscapes, and output distributions were visualized.
- A quantum kernel-based classification approach for MaxCut was also briefly explored.

---

## 🧪 Features

- Implementation of QAOA with tunable depth \( p \)
- Graph encoding, Hamiltonian construction, and parameter optimization
- Energy landscape visualization for different QAOA depths
- Comparative analysis using both **PennyLane** and **Qiskit**
- Basic reproduction of PennyLane QAOA results in OpenQASM
- Bonus: Quantum embedding kernel (QEK) method for MaxCut as classification

---


## 🚀 Getting Started
Dependencies include:
- `pennylane`
- `qiskit`
- `matplotlib`
- `numpy`
- `networkx`

### ▶️ Run QAOA

You can modify the graph structure and QAOA depth in the script to reproduce various experiments.

---

## 🧠 Key Findings

- QAOA achieves optimal or near-optimal MaxCut solutions on small graphs.
- Increasing QAOA depth \( p \) improves performance but adds circuit complexity.
- Energy landscape visualizations provide insight into parameter optimization.
- Careful attention must be paid when reproducing QAOA results across frameworks due to qubit-node mapping (especially in OpenQASM).

---

## 📄 Report

For detailed methodology, results, and discussion, please see the full report:  
📎 `/13_QAOA.pdf`

---

## 🧾 References

Key software frameworks used:
- [PennyLane](https://github.com/PennyLaneAI/pennylane)
- [Qiskit](https://github.com/Qiskit)
- [OpenQASM](https://github.com/Qiskit/openqasm)

---

## 🙏 Acknowledgments

The author would like to acknowledge the development teams and open-source contributors of **PennyLane**, **Qiskit**, and **OpenQASM** for providing the tools and frameworks that enabled this research.
