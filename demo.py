import time

from decomp_heuristics import *
from flops import rw_simulate_flops, quimb_flops
from gf2 import REF
from graph import adjacency_matrix, rank_width, rank_score_flops, rank_score_square
from rw_simulate import simulate_circuit
from quimb_helpers import quimb_amplitude
from zx_helpers import circuit2graph


def greedy_inc_decomposition(g: zx.graph.base.BaseGraph):
    n = g.num_vertices()
    vert = list(g.vertices())
    mat = adjacency_matrix(g, vert, vert)
    ref_init = REF(mat)
    ref_init.take(0)
    decomp = [vert[0], ref_init, {0}]

    def insert_leaf(data, leaf, leaves_to_attach):
        if data[2] == leaves_to_attach:
            child_ref = REF(mat)
            child_ref.take(leaf)
            new_ref = deepcopy(data[1])
            new_ref.take(leaf)
            new_data = [data, (leaf, child_ref, {leaf})], new_ref, data[2] | {leaf}
            return new_data
        child1_leaves, child2_leaves = data[0][0][2], data[0][1][2]
        if child1_leaves & leaves_to_attach:
            new_child = insert_leaf(data[0][0], leaf, leaves_to_attach)
            new_ref = deepcopy(data[1])
            new_ref.take(leaf)
            new_data = [new_child, data[0][1]], new_ref, data[2] | {leaf}
            return new_data
        else:
            new_child = insert_leaf(data[0][1], leaf, leaves_to_attach)
            new_ref = deepcopy(data[1])
            new_ref.take(leaf)
            new_data = [data[0][0], new_child], new_ref, data[2] | {leaf}
            return new_data

    def find_best_extension(root, elem, depth):
        best_score = n ** 2
        best_decomp = None
        for j in elem[1].pivot_cols:
            if j in root[2]:
                continue
            new_root = insert_leaf(root, j, elem[2])
            # score = rank_width(new_root, g, calc_rs=False)
            # score = rank_score_flops(new_root, g, calc_rs=False)
            # score = rank_score_square(new_root, g, calc_rs=False)
            score = rank_score_square(new_root, j, calc_rs=False) + 3 * rank_width(new_root, j, calc_rs=False)
            if score < best_score:
                best_score = score
                best_decomp = new_root
        if isinstance(elem[0], int):
            return best_score, best_decomp
        if depth < 10:
            best_score1, best_decomp1 = find_best_extension(root, elem[0][0], depth + 1)
            best_score2, best_decomp2 = find_best_extension(root, elem[0][1], depth + 1)
            if best_score1 < best_score:
                best_score = best_score1
                best_decomp = best_decomp1
            if best_score2 < best_score:
                best_score = best_score2
                best_decomp = best_decomp2
        return best_score, best_decomp

    for _ in range(n - 1):
        score, decomp = find_best_extension(decomp, decomp, 0)
        print(rank_score_flops(decomp, g, calc_rs=False))
    return decomp


circ = zx.Circuit.load('circuits/special/ham15-low.qc')
# circ = zx.generate.CNOT_HAD_PHASE_circuit(10, 2000)
state, effect = 'T' * circ.qubits, 'T' * circ.qubits
# print(f'Our flops (flow): {rw_simulate_flops(circ, state, effect, opt="flow")}')
# print(f'Our flops (linear): {rw_simulate_flops(circ, state, effect, opt="linear")}')
# print(f'Our flops (greedy): {rw_simulate_flops(circ, state, effect, opt="greedy")}')
# print(f'Our flops (dp): {rw_simulate_flops(circ, state, effect, opt="dp")}')
# print(f'Our flops (auto): {rw_simulate_flops(circ, state, effect, opt="auto")}')
# print(f'Quimb flops (auto): {quimb_flops(circ, state, effect, optimize="auto", initial=True)}')
# print(f'Quimb flops (auto-hq): {quimb_flops(circ, state, effect, optimize="auto-hq", initial=True)}')

g = circuit2graph(circ, state, effect).copy(backend='quizx-vec')
g.apply_state('0' * circ.qubits)
g.apply_effect('0' * circ.qubits)
quizx.full_simp(g)
print('n_vertices =', g.num_vertices())
print('n_qubits =', circ.qubits)

t0 = time.time()
decomp0 = auto_decomposition(g)
t1 = time.time()
print(f'auto decomposition computed in {t1 - t0:.3f} sec')
print(f'auto decomposition has width {rank_width(decomp0, g)} and score {rank_score_flops(decomp0, g)}')

# t0 = time.time()
# decomp1 = linear_decomposition(g)
# t1 = time.time()
# print(f'linear decomposition computed in {t1 - t0:.3f} sec')
# print(f'linear decomposition has width {rank_width(decomp1, g)} and score {rank_score_flops(decomp1, g)}')
#
t0 = time.time()
decomp2 = greedy_decomposition(g)
t1 = time.time()
print(f'greedy decomposition computed in {t1 - t0:.3f} sec')
print(f'greedy decomposition has width {rank_width(decomp2, g)} and score {rank_score_flops(decomp2, g)}')

# t0 = time.time()
# decomp3 = linear_dp_decomposition(g)
# t1 = time.time()
# print(f'linear-dp decomposition computed in {t1 - t0:.3f} sec')
# print(f'linear-dp decomposition has width {rank_width(decomp3, g)} and score {rank_score_flops(decomp3, g)}')

# t0 = time.time()
# decomp4 = greedy_inc_decomposition(g)
# t1 = time.time()
# print(f'greedy-inc decomposition computed in {t1 - t0:.3f} sec')
# print(f'greedy-inc decomposition has width {rank_width(decomp4, g, calc_rs=False)} and score {rank_score_flops(decomp4, g, calc_rs=False)}')
