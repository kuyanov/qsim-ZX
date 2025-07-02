import pyzx as zx
from pyzx.gflow import gflow


n_qubits = 10
n_gates = 20
c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=n_qubits, depth=n_gates)
g = c.to_graph()
# g.apply_state('0' * n_qubits)
# g.apply_effect('0' * n_qubits)
zx.clifford_simp(g)
print(gflow(g))
zx.draw(g, labels=True)
