from tqdm import trange
from flops import *

circuit_dir = 'circuits/special'
initial = True
fs = sorted(os.listdir(circuit_dir))
for i in trange(len(fs)):
    f = fs[i]
    circ = zx.Circuit.load(os.path.join(circuit_dir, f))
    print(f'Circuit {f} with {circ.qubits} qubits and {len(circ.gates)} gates')
    state, effect = 'T' * circ.qubits, 'T' * circ.qubits
    flops_rw = rw_simulate_flops(circ, state, effect, seed=1)
    print(f'Our flops: {flops_rw}')
    quimb_flops_auto = quimb_flops(circ, state, effect, 'auto', initial)
    print(f'Quimb flops (auto): {quimb_flops_auto}')
    quimb_flops_auto_hq = quimb_flops(circ, state, effect, 'auto-hq', initial)
    print(f'Quimb flops (auto-hq): {quimb_flops_auto_hq}')
    quimb_flops_greedy = quimb_flops(circ, state, effect, 'greedy', initial)
    print(f'Quimb flops (greedy): {quimb_flops_greedy}')
