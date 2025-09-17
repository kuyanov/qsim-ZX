import pyzx as zx
import os
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import trange
from flops import rw_simulate_flops, quimb_flops

if __name__ == '__main__':
    circuit_dir = 'circuits/special'
    batch1 = ['tof_3', 'tof_4', 'tof_5', 'tof_10',
              'barenco_tof_3.qasm', 'barenco_tof_4.qasm', 'barenco_tof_5.qasm',
              'qft_4.qasm', 'qft_8', 'hwb6.qc']
    batch2 = ['adder_8', 'rc_adder_6', 'vbe_adder_3', 'mod_mult_55', 'mod_red_21',
              'gf2^4_mult', 'csla_mux_3_original', 'csum_mux_9_corrected',
              'qcla_com_7', 'ham15-low.qc']

    data = {
        'circuit': [],
        'flops': [],
        'strategy': [],
    }

    batch = batch2
    for i in trange(len(batch)):
        circ = zx.Circuit.load(os.path.join(circuit_dir, batch[i]))
        state, effect = 'T' * circ.qubits, 'T' * circ.qubits
        circ_name = os.path.splitext(batch[i])[0]

        flops_quimb_init_auto = quimb_flops(circ, state, effect, optimize='auto', initial=True)
        data['circuit'].append(circ_name)
        data['flops'].append(flops_quimb_init_auto)
        data['strategy'].append('quimb_init (auto)')

        flops_quimb_init_auto_hq = quimb_flops(circ, state, effect, optimize='auto-hq', initial=True)
        data['circuit'].append(circ_name)
        data['flops'].append(flops_quimb_init_auto_hq)
        data['strategy'].append('quimb_init (auto-hq)')

        flops_quimb_auto = quimb_flops(circ, state, effect, optimize='auto', initial=False)
        data['circuit'].append(circ_name)
        data['flops'].append(flops_quimb_auto)
        data['strategy'].append('quimb (auto)')

        flops_quimb_auto_hq = quimb_flops(circ, state, effect, optimize='auto-hq', initial=False)
        data['circuit'].append(circ_name)
        data['flops'].append(flops_quimb_auto_hq)
        data['strategy'].append('quimb (auto-hq)')

        flops_rw_flow = rw_simulate_flops(circ, state, effect, opt='flow')
        data['circuit'].append(circ_name)
        data['flops'].append(flops_rw_flow)
        data['strategy'].append('rank-width (flow)')

        flops_rw_linear = rw_simulate_flops(circ, state, effect, opt='linear')
        data['circuit'].append(circ_name)
        data['flops'].append(flops_rw_linear)
        data['strategy'].append('rank-width (linear)')

        flops_rw_greedy = rw_simulate_flops(circ, state, effect, opt='greedy')
        data['circuit'].append(circ_name)
        data['flops'].append(flops_rw_greedy)
        data['strategy'].append('rank-width (greedy)')

        flops_rw_dp = rw_simulate_flops(circ, state, effect, opt='dp')
        data['circuit'].append(circ_name)
        data['flops'].append(flops_rw_dp)
        data['strategy'].append('rank-width (dp)')

        flops_rw_auto = rw_simulate_flops(circ, state, effect, opt='auto')
        data['circuit'].append(circ_name)
        data['flops'].append(flops_rw_auto)
        data['strategy'].append('rank-width (auto)')

    sns.barplot(data, x='circuit', y='flops', hue='strategy', palette='bright')
    plt.xticks(rotation=25)
    plt.yscale('log')
    plt.xlabel('')
    plt.show()
