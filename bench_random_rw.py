from tqdm import trange

from rw_simulate import *

n_qubits_gates = [
    (10, 50), (10, 100), (10, 200), (10, 300),
    (15, 75), (15, 150), (15, 250), (15, 350),
    (20, 100), (20, 200), (20, 300), (20, 400),
    (25, 125), (25, 250), (25, 350), (25, 450),
]

for i in trange(len(n_qubits_gates)):
    n_qubits, n_gates = n_qubits_gates[i]
    circ = zx.generate.CNOT_HAD_PHASE_circuit(qubits=n_qubits, depth=n_gates)
    state, effect = '0' * n_qubits, '0' * n_qubits
    g = circ.to_graph()
    zx.full_reduce(g)
    decomp = pauli_flow_decomposition(g)
    g.apply_state(state)
    g.apply_effect(effect)
    zx.clifford_simp(g)
    g2 = g.copy(backend="quizx-vec")
    if g2.num_vertices() == 0:
        print(f'Q={n_qubits} G={n_gates} NONE')
    decomp = sub_decomposition(decomp, list(g.vertices()), list(g2.vertices()))
    init_decomp = quizx.DecompTree.from_list(decomp)
    init_score = init_decomp.rankwidth_score(g2, kind='flops')
    init_rw = init_decomp.rankwidth(g2)
    ann = quizx.RankwidthAnnealer(g2, init_decomp=init_decomp, score_kind='square',
                                  seed=1, init_temp=2, cooling_rate=0.99, min_temp=1e-3)
    final_decomp = ann.run()
    final_score = final_decomp.rankwidth_score(g2, kind='flops')
    final_rw = final_decomp.rankwidth(g2)
    print(f'Q={n_qubits} G={n_gates}, '
          f'score: {init_score:.3f} -> {final_score:.3f}, '
          f'rw: {init_rw} -> {final_rw}')
