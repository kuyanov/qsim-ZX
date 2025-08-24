import os

from tqdm import trange
from flops import *

circuit_dir = 'circuits/all'
initial = True
fs = sorted(os.listdir(circuit_dir))
for i in trange(len(fs)):
    f = fs[i]
    circ = zx.Circuit.load(os.path.join(circuit_dir, f))
    state, effect = '-' * circ.qubits, '-' * circ.qubits
    flops_rw = rw_simulate_flops(circ, state, effect)
    quimb_flops_auto = quimb_flops(circ, state, effect, 'auto', initial)
    quimb_flops_auto_hq = quimb_flops(circ, state, effect, 'auto-hq', initial)
    quimb_flops_greedy = quimb_flops(circ, state, effect, 'greedy', initial)
    print(f'{f} Q={circ.qubits} G={len(circ.gates)} '
          f'flops_auto={quimb_flops_auto} '
          f'flops_auto_hq={quimb_flops_auto_hq} '
          f'flops_greedy={quimb_flops_greedy} '
          f'flops_rw={flops_rw}')
