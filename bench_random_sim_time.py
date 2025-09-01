import time

import matplotlib.pyplot as plt
import seaborn as sns

from quimb_helpers import *
from rw_simulate import *

n_qubits = 20
ns_gates = list(range(50, 501, 25))
n_iter = 5

data_rw = {
    'G': [],
    'time': [],
}
data_quimb = {
    'G': [],
    'time': [],
}
for i, n_gates in enumerate(ns_gates):
    for j in range(n_iter):
        circ = zx.generate.CNOT_HAD_PHASE_circuit(n_qubits, n_gates)
        state, effect = '0' * circ.qubits, '0' * circ.qubits

        t0 = time.time()
        res_rw = simulate_circuit(circ, state, effect, iterations=0)
        t1 = time.time()
        data_rw['G'].append(n_gates)
        data_rw['time'].append(t1 - t0)

        t0 = time.time()
        res_quimb = quimb_amplitude(circuit2quimb(circ), state, effect)
        t1 = time.time()
        data_quimb['G'].append(n_gates)
        data_quimb['time'].append(t1 - t0)

sns.pointplot(data_rw, x='G', y='time', label='rank-width', log_scale=True)
sns.pointplot(data_quimb, x='G', y='time', label='quimb', color='black', log_scale=True)
# plt.legend()
plt.show()
