from tqdm import trange

from rw_simulate import *

circuit_dir = 'circuits/special'
fs = sorted(os.listdir(circuit_dir))
for i in trange(len(fs)):
    f = fs[i]
    circ = zx.Circuit.load(os.path.join(circuit_dir, f))
    print(f'Circuit {f} with {circ.qubits} qubits and {len(circ.gates)} gates')
    state, effect = 'T' * circ.qubits, 'T' * circ.qubits
    g, decomp = initial_decomposition(circ, state, effect, verbose=True)
    g, decomp = improve_decomposition(g, decomp, verbose=True, score_kind='square',
                                      seed=1, cooling_rate=0.99, init_temp=10, min_temp=1e-3)
