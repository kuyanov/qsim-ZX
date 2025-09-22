import pyzx as zx
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from matplotlib.patches import Patch
from tqdm import trange
from flops import rw_simulate_flops, quimb_flops


def per_circuit_barplot(data, **kwargs):
    order = data.groupby('strategy')['flops'].sum().sort_values().index
    ax = sns.barplot(
        data=data,
        x='strategy',
        y='flops',
        order=order,
        hue='strategy',
        palette='bright',
        **kwargs
    )
    ax.set_xticks([])
    ax.set_title('')
    ax.set_xlabel(data['circuit'].iloc[0])
    ax.set_yscale('log')


if __name__ == '__main__':
    circuit_dir = 'circuits/small'
    batch = sorted(os.listdir(circuit_dir))
    data = {
        'circuit': [],
        'flops': [],
        'strategy': [],
    }

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

        # flops_quimb_auto = quimb_flops(circ, state, effect, optimize='auto', initial=False)
        # data['circuit'].append(circ_name)
        # data['flops'].append(flops_quimb_auto)
        # data['strategy'].append('quimb (auto)')
        #
        # flops_quimb_auto_hq = quimb_flops(circ, state, effect, optimize='auto-hq', initial=False)
        # data['circuit'].append(circ_name)
        # data['flops'].append(flops_quimb_auto_hq)
        # data['strategy'].append('quimb (auto-hq)')

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

    df = pd.DataFrame(data)
    g = sns.FacetGrid(df, col='circuit', col_wrap=5, sharex=False)
    g.map_dataframe(per_circuit_barplot)

    strategies = df['strategy'].unique()
    palette_list = sns.color_palette('bright', n_colors=len(strategies))
    palette = dict(zip(strategies, palette_list))
    handles = [Patch(facecolor=palette[s], label=s) for s in strategies]
    g.figure.legend(handles=handles, ncol=3)

    plt.tight_layout()
    plt.savefig('results/bench-special/small_flops_full.png')
