import pyzx as zx
import time
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import trange
from quimb_helpers import quimb_amplitude
from rw_simulate import simulate_circuit

if __name__ == '__main__':
    n_qubits = 10
    ns_gates = list(range(100, 501, 25))
    repeat = 3

    data = {
        'n_gates': [],
        'time': [],
        'strategy': [],
    }

    for i in trange(len(ns_gates)):
        n_gates = ns_gates[i]
        for j in range(repeat):
            circ = zx.generate.CNOT_HAD_PHASE_circuit(n_qubits, n_gates)
            state, effect = '0' * circ.qubits, '0' * circ.qubits

            t0 = time.time()
            res_quimb_init_auto = quimb_amplitude(circ, state, effect, optimize='auto', initial=True)
            t1 = time.time()
            data['n_gates'].append(n_gates)
            data['time'].append(t1 - t0)
            data['strategy'].append('quimb_init (auto)')

            t0 = time.time()
            res_quimb_init_auto_hq = quimb_amplitude(circ, state, effect, optimize='auto-hq', initial=True)
            t1 = time.time()
            data['n_gates'].append(n_gates)
            data['time'].append(t1 - t0)
            data['strategy'].append('quimb_init (auto-hq)')

            # t0 = time.time()
            # res_quimb_auto = quimb_amplitude(circ, state, effect, optimize='auto', initial=False)
            # t1 = time.time()
            # data['n_gates'].append(n_gates)
            # data['time'].append(t1 - t0)
            # data['strategy'].append('quimb (auto)')
            #
            # t0 = time.time()
            # res_quimb_auto_hq = quimb_amplitude(circ, state, effect, optimize='auto-hq', initial=False)
            # t1 = time.time()
            # data['n_gates'].append(n_gates)
            # data['time'].append(t1 - t0)
            # data['strategy'].append('quimb (auto-hq)')

            t0 = time.time()
            res_rw_flow = simulate_circuit(circ, state, effect, opt='flow')
            t1 = time.time()
            data['n_gates'].append(n_gates)
            data['time'].append(t1 - t0)
            data['strategy'].append('rank-width (flow)')

            t0 = time.time()
            res_rw_linear = simulate_circuit(circ, state, effect, opt='linear')
            t1 = time.time()
            data['n_gates'].append(n_gates)
            data['time'].append(t1 - t0)
            data['strategy'].append('rank-width (linear)')

            t0 = time.time()
            res_rw_greedy = simulate_circuit(circ, state, effect, opt='greedy')
            t1 = time.time()
            data['n_gates'].append(n_gates)
            data['time'].append(t1 - t0)
            data['strategy'].append('rank-width (greedy)')

            t0 = time.time()
            res_rw_dp = simulate_circuit(circ, state, effect, opt='dp')
            t1 = time.time()
            data['n_gates'].append(n_gates)
            data['time'].append(t1 - t0)
            data['strategy'].append('rank-width (dp)')

            t0 = time.time()
            res_rw_auto = simulate_circuit(circ, state, effect, opt='auto')
            t1 = time.time()
            data['n_gates'].append(n_gates)
            data['time'].append(t1 - t0)
            data['strategy'].append('rank-width (auto)')

    sns.pointplot(data, x='n_gates', y='time', hue='strategy', palette='bright')
    plt.yscale('log')
    plt.show()
