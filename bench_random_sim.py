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
    state, effect = '0' * n_qubits, '0' * n_qubits
    flops_rw = rw_simulate_flops(circ, state, effect)
    quimb_flops_auto = quimb_flops(circ, state, effect, 'auto', initial)
    quimb_flops_auto_hq = quimb_flops(circ, state, effect, 'auto-hq', initial)
    quimb_flops_greedy = quimb_flops(circ, state, effect, 'greedy', initial)
    print(f'Q={circ.qubits} G={len(circ.gates)} '
          f'flops_auto={quimb_flops_auto} '
          f'flops_auto_hq={quimb_flops_auto_hq} '
          f'flops_greedy={quimb_flops_greedy} '
          f'flops_rw={flops_rw}')
