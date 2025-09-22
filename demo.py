import time

from decomp_heuristics import *
from flops import rw_simulate_flops, quimb_flops
from graph import rank_width, rank_score_flops
from zx_helpers import circuit2graph

circ = zx.Circuit.load('circuits/small/ham15-low.qasm')
# circ = zx.generate.CNOT_HAD_PHASE_circuit(10, 1000)
state, effect = 'T' * circ.qubits, 'T' * circ.qubits
print(f'Our flops (flow): {rw_simulate_flops(circ, state, effect, opt="flow")}')
print(f'Our flops (linear): {rw_simulate_flops(circ, state, effect, opt="linear")}')
print(f'Our flops (greedy): {rw_simulate_flops(circ, state, effect, opt="greedy")}')
print(f'Our flops (dp): {rw_simulate_flops(circ, state, effect, opt="dp")}')
print(f'Our flops (auto): {rw_simulate_flops(circ, state, effect, opt="auto")}')
print(f'Quimb_init flops (auto): {quimb_flops(circ, state, effect, optimize="auto", initial=True)}')
print(f'Quimb_init flops (auto-hq): {quimb_flops(circ, state, effect, optimize="auto-hq", initial=True)}')
# print(f'Quimb flops (auto): {quimb_flops(circ, state, effect, optimize="auto", initial=False)}')

# g = circuit2graph(circ, state, effect)
# g.apply_state('0' * circ.qubits)
# g.apply_effect('0' * circ.qubits)
# zx.full_reduce(g)
# print('n_vertices =', g.num_vertices())
# print('n_qubits =', circ.qubits)
#
# t0 = time.time()
# decomp_auto = auto_decomposition(g)
# t1 = time.time()
# print(f'auto decomposition computed in {t1 - t0:.3f} sec')
# print(f'auto decomposition has width {rank_width(decomp_auto, g)} and score {rank_score_flops(decomp_auto, g)}')
#
# t0 = time.time()
# decomp_linear = linear_decomposition(g)
# t1 = time.time()
# print(f'linear decomposition computed in {t1 - t0:.3f} sec')
# print(f'linear decomposition has width {rank_width(decomp_linear, g)} and score {rank_score_flops(decomp_linear, g)}')
#
# t0 = time.time()
# decomp_greedy = greedy_decomposition(g)
# t1 = time.time()
# print(f'greedy decomposition computed in {t1 - t0:.3f} sec')
# print(f'greedy decomposition has width {rank_width(decomp_greedy, g)} and score {rank_score_flops(decomp_greedy, g)}')
#
# t0 = time.time()
# decomp_dp = dp_decomposition(g)
# t1 = time.time()
# print(f'dp decomposition computed in {t1 - t0:.3f} sec')
# print(f'dp decomposition has width {rank_width(decomp_dp, g)} and score {rank_score_flops(decomp_dp, g)}')
