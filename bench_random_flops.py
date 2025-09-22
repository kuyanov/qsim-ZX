import pyzx as zx
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import trange
from flops import rw_simulate_flops, quimb_flops

if __name__ == '__main__':
    n_qubits = 10
    ns_gates = list(range(100, 801, 25))
    repeat = 3
    data = {
        'n_gates': [],
        'flops': [],
        'strategy': [],
    }

    for i in trange(len(ns_gates)):
        n_gates = ns_gates[i]
        for j in range(repeat):
            circ = zx.generate.CNOT_HAD_PHASE_circuit(n_qubits, n_gates)
            state, effect = '0' * circ.qubits, '0' * circ.qubits

            flops_quimb_init_auto = quimb_flops(circ, state, effect, optimize='auto', initial=True)
            data['n_gates'].append(n_gates)
            data['flops'].append(flops_quimb_init_auto)
            data['strategy'].append('quimb_init (auto)')

            flops_quimb_init_auto_hq = quimb_flops(circ, state, effect, optimize='auto-hq', initial=True)
            data['n_gates'].append(n_gates)
            data['flops'].append(flops_quimb_init_auto_hq)
            data['strategy'].append('quimb_init (auto-hq)')

            # flops_quimb_auto = quimb_flops(circ, state, effect, optimize='auto', initial=False)
            # data['n_gates'].append(n_gates)
            # data['flops'].append(flops_quimb_auto)
            # data['strategy'].append('quimb (auto)')
            #
            # flops_quimb_auto_hq = quimb_flops(circ, state, effect, optimize='auto-hq', initial=False)
            # data['n_gates'].append(n_gates)
            # data['flops'].append(flops_quimb_auto_hq)
            # data['strategy'].append('quimb (auto-hq)')

            flops_rw_flow = rw_simulate_flops(circ, state, effect, opt='flow')
            data['n_gates'].append(n_gates)
            data['flops'].append(flops_rw_flow)
            data['strategy'].append('rank-width (flow)')

            flops_rw_linear = rw_simulate_flops(circ, state, effect, opt='linear')
            data['n_gates'].append(n_gates)
            data['flops'].append(flops_rw_linear)
            data['strategy'].append('rank-width (linear)')

            flops_rw_greedy = rw_simulate_flops(circ, state, effect, opt='greedy')
            data['n_gates'].append(n_gates)
            data['flops'].append(flops_rw_greedy)
            data['strategy'].append('rank-width (greedy)')

            flops_rw_dp = rw_simulate_flops(circ, state, effect, opt='dp')
            data['n_gates'].append(n_gates)
            data['flops'].append(flops_rw_dp)
            data['strategy'].append('rank-width (dp)')

            flops_rw_auto = rw_simulate_flops(circ, state, effect, opt='auto')
            data['n_gates'].append(n_gates)
            data['flops'].append(flops_rw_auto)
            data['strategy'].append('rank-width (auto)')

    plt.figure(figsize=(14, 8))
    sns.pointplot(data, x='n_gates', y='flops', hue='strategy', palette='bright')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(f'results/bench-random/CNOT_H_T_Q{n_qubits}_flops_full.png')
