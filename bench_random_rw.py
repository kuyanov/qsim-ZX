from tqdm import trange

from rw_simulate import *

n_qubits_gates = [
    (10, 50), (10, 100), (10, 200), (10, 300),
    (15, 75), (15, 150), (15, 250), (15, 350),
    (20, 100), (20, 200), (20, 300), (20, 400),
    (25, 125), (25, 250), (25, 350), (25, 450),
]
for i in trange(len(n_qubits_gates)):
    n_qubits, n_gates = n_qubits_gates[i]
    circ = zx.generate.CNOT_HAD_PHASE_circuit(qubits=n_qubits, depth=n_gates)
    print(f'Random circuit with {circ.qubits} qubits and {len(circ.gates)} gates')
    state, effect = '-' * n_qubits, '-' * n_qubits
    g, decomp = initial_decomposition(circ, state, effect, verbose=True)
    g, decomp = improve_decomposition(g, decomp, verbose=True, seed=1)
