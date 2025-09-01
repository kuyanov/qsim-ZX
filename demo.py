import time

import numpy as np

from flops import *
from quimb_helpers import *
from rw_simulate import *

# c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=n_qubits, depth=n_gates)
# print(simulate_circuit(c, state, effect, verbose=True))
# quizx.full_simp(g)
# print(g.scalar.to_number())
# print(zx.tensorfy(g))
# print(g.num_vertices(), g.num_edges())

# circ = zx.Circuit.load('circuits/special/mod_mult_55')
circ = zx.generate.CNOT_HAD_PHASE_circuit(20, 300)
state, effect = '0' * circ.qubits, '0' * circ.qubits

t0 = time.time()
res_rw = simulate_circuit(circ, state, effect, iterations=0)
t1 = time.time()
print(f'our time: {t1 - t0:.3f} sec')
t0 = time.time()
res_quimb = quimb_amplitude(circuit2quimb(circ), state, effect)
t1 = time.time()
print(f'quimb time: {t1 - t0:.3f} sec')

t0 = time.time()
g = circuit2graph(circ, state, effect)
g.apply_state('0' * circ.qubits)
g.apply_effect('0' * circ.qubits)
res_tfy = zx.tensorfy(g)
t1 = time.time()
print(f'tensorfy time: {t1 - t0:.3f} sec')

# circuit_name = 'circuits/special/csla_mux_3_original'
# circuit_name = 'circuits/special/csum_mux_9_corrected'
# circ = zx.Circuit.load(circuit_name)
# state, effect = 'T' * circ.qubits, 'T' * circ.qubits
# qcirc = circuit2quimb(circ)
# print('quimb_flops:', quimb_flops(circ, state, effect, 'auto-hq', True))
# print(quimb_amplitude(qcirc, state, effect))
# g = circuit2graph(circ, state, effect)
# g.apply_state('0' * circ.qubits)
# g.apply_effect('0' * circ.qubits)
# # print(zx.tensorfy(g))
# zx.full_reduce(g)
# zx.draw(g)
