import csv
import numpy as np
import os
import pyzx as zx
from matplotlib import pyplot as plt

from rank_width import rw_decomposition


def gen_reduced_diagram(n_qubits, n_gates):
    c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=n_qubits, depth=n_gates)
    g = c.to_graph()
    g.apply_state('0' * n_qubits)
    g.apply_effect('0' * n_qubits)
    zx.simplify.full_reduce(g)
    return g


if __name__ == "__main__":
    output_dir = 'results/random-rw-benchmark'
    os.makedirs(output_dir, exist_ok=True)

    timeout_s = 10
    num_iter = 10
    n_qubits_arr = [10]
    n_gates_arr = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    rw_median = [[-1] * len(n_gates_arr) for _ in range(len(n_qubits_arr))]
    for row, n_qubits in enumerate(n_qubits_arr):
        for col, n_gates in enumerate(n_gates_arr):
            ranks = []
            while len(ranks) < num_iter:
                g = gen_reduced_diagram(n_qubits, n_gates)
                tree_edges = rw_decomposition(g, timeout_s=timeout_s)
                if tree_edges is not None:
                    r = max([0] + [r for u, v, r in tree_edges])
                    ranks.append(r)

            rw_median[row][col] = int(np.round(np.median(ranks)))
            plt.hist(ranks)
            plt.title(f"Rank-width distribution for N={n_qubits}, G={n_gates}, T={timeout_s}s")
            plt.xlabel('r')
            fig_path = f'{output_dir}/rw_N{n_qubits}_G{n_gates}_{timeout_s}s.png'
            print(f'saving: {fig_path}')
            plt.savefig(fig_path)
            plt.close()

    with open(f'{output_dir}/rw_median_N10_{timeout_s}s.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['N\G'] + n_gates_arr)
        for n_qubits, row in zip(n_qubits_arr, rw_median):
            writer.writerow([n_qubits] + row)
