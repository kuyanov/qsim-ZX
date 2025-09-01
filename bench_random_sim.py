from tqdm import trange

from flops import *


n_qubits_gates = [
    (10, 50), (10, 100), (10, 200), (10, 300),
    (15, 75), (15, 150), (15, 250), (15, 350),
    (20, 100), (20, 200), (20, 300), (20, 400),
    (25, 125), (25, 250), (25, 350), (25, 450),
]
initial = True
for i in trange(len(n_qubits_gates)):
    n_qubits, n_gates = n_qubits_gates[i]
    circ = zx.generate.CNOT_HAD_PHASE_circuit(qubits=n_qubits, depth=n_gates)
    print(f'Random circuit with {circ.qubits} qubits and {len(circ.gates)} gates')
    state, effect = '-' * circ.qubits, '-' * circ.qubits
    flops_rw = rw_simulate_flops(circ, state, effect, seed=1)
    print(f'Our flops: {flops_rw}')
    quimb_flops_auto = quimb_flops(circ, state, effect, 'auto', initial)
    print(f'Quimb flops (auto): {quimb_flops_auto}')
    quimb_flops_auto_hq = quimb_flops(circ, state, effect, 'auto-hq', initial)
    print(f'Quimb flops (auto-hq): {quimb_flops_auto_hq}')
    quimb_flops_greedy = quimb_flops(circ, state, effect, 'greedy', initial)
    print(f'Quimb flops (greedy): {quimb_flops_greedy}')