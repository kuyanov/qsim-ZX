import os
from tqdm import trange

from rw_simulate import *


circuit_dir = 'circuits/all'
fs = sorted(os.listdir(circuit_dir))
for i in trange(len(fs)):
    f = fs[i]
    circ = zx.Circuit.load(os.path.join(circuit_dir, f))
    state, effect = '-' * circ.qubits, '-' * circ.qubits
    g = circ.to_graph()
    zx.full_reduce(g)

    if g.num_vertices() > 300:
        continue
    decomp = pauli_flow_decomposition(g)

    g.apply_state(state)
    g.apply_effect(effect)
    zx.clifford_simp(g)
    if g.num_vertices() > 150:
        continue
    g2 = g.copy(backend="quizx-vec")
    decomp = sub_decomposition(decomp, list(g.vertices()), list(g2.vertices()))
    init_decomp = quizx.DecompTree.from_list(decomp)
    init_score = init_decomp.rankwidth_score(g2, kind='flops')
    init_rw = init_decomp.rankwidth(g2)

    ann = quizx.RankwidthAnnealer(g2, init_decomp=init_decomp, score_kind='square',
                                  seed=1, init_temp=2, cooling_rate=0.99, min_temp=1e-3)
    final_decomp = ann.run()
    final_score = final_decomp.rankwidth_score(g2, kind='flops')
    final_rw = final_decomp.rankwidth(g2)
    print(f'{f} Q={circ.qubits} G={len(circ.gates)}, score: {init_score:.3f} -> {final_score:.3f}, rw: {init_rw} -> {final_rw}')
