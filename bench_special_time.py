import pyzx as zx
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import trange
from quimb_helpers import quimb_amplitude
from rw_simulate import simulate_circuit

if __name__ == '__main__':
    circuit_dir = 'circuits/special'
    batch1 = ['tof_3', 'tof_4', 'tof_5', 'tof_10',
              'barenco_tof_3.qasm', 'barenco_tof_4.qasm', 'barenco_tof_5.qasm',
              'qft_4.qasm', 'qft_8', 'hwb6.qc']
    batch2 = ['adder_8', 'rc_adder_6', 'vbe_adder_3', 'mod_mult_55', 'mod_red_21',
              'gf2^4_mult', 'csla_mux_3_original', 'qcla_com_7']

    data = {
        'circuit': [],
        'time': [],
        'strategy': [],
    }

    batch = batch2
    for i in trange(len(batch)):
        circ = zx.Circuit.load(os.path.join(circuit_dir, batch[i]))
        state, effect = 'T' * circ.qubits, 'T' * circ.qubits
        circ_name = os.path.splitext(batch[i])[0]

        t0 = time.time()
        res_quimb_init_auto = quimb_amplitude(circ, state, effect, optimize='auto', initial=True)
        t1 = time.time()
        data['circuit'].append(circ_name)
        data['time'].append(t1 - t0)
        data['strategy'].append('quimb_init (auto)')

        t0 = time.time()
        res_quimb_init_auto_hq = quimb_amplitude(circ, state, effect, optimize='auto-hq', initial=True)
        t1 = time.time()
        data['circuit'].append(circ_name)
        data['time'].append(t1 - t0)
        data['strategy'].append('quimb_init (auto-hq)')

        t0 = time.time()
        res_quimb_auto = quimb_amplitude(circ, state, effect, optimize='auto', initial=False)
        t1 = time.time()
        data['circuit'].append(circ_name)
        data['time'].append(t1 - t0)
        data['strategy'].append('quimb (auto)')

        t0 = time.time()
        res_quimb_auto_hq = quimb_amplitude(circ, state, effect, optimize='auto-hq', initial=False)
        t1 = time.time()
        data['circuit'].append(circ_name)
        data['time'].append(t1 - t0)
        data['strategy'].append('quimb (auto-hq)')

        t0 = time.time()
        res_rw_flow = simulate_circuit(circ, state, effect, opt='flow')
        t1 = time.time()
        data['circuit'].append(circ_name)
        data['time'].append(t1 - t0)
        data['strategy'].append('rank-width (flow)')

        t0 = time.time()
        res_rw_linear = simulate_circuit(circ, state, effect, opt='linear')
        t1 = time.time()
        data['circuit'].append(circ_name)
        data['time'].append(t1 - t0)
        data['strategy'].append('rank-width (linear)')

        t0 = time.time()
        res_rw_greedy = simulate_circuit(circ, state, effect, opt='greedy')
        t1 = time.time()
        data['circuit'].append(circ_name)
        data['time'].append(t1 - t0)
        data['strategy'].append('rank-width (greedy)')

        t0 = time.time()
        res_rw_dp = simulate_circuit(circ, state, effect, opt='dp')
        t1 = time.time()
        data['circuit'].append(circ_name)
        data['time'].append(t1 - t0)
        data['strategy'].append('rank-width (dp)')

        t0 = time.time()
        res_rw_auto = simulate_circuit(circ, state, effect, opt='auto')
        t1 = time.time()
        data['circuit'].append(circ_name)
        data['time'].append(t1 - t0)
        data['strategy'].append('rank-width (auto)')

    sns.barplot(data, x='circuit', y='time', hue='strategy', palette='bright')
    plt.xticks(rotation=25)
    plt.yscale('log')
    plt.xlabel('')
    plt.show()
