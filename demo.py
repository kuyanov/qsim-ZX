import pyzx as zx

from rw_simulate import simulate_circuit

n_qubits = 15
n_gates = 200
c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=n_qubits, depth=n_gates)

print(simulate_circuit(c, '0' * n_qubits, '0' * n_qubits))
