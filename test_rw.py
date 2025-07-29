import numpy as np
import pyzx as zx
import random
from galois import GF2

from gf2 import rank_factorize, generalized_inverse
from graph import adjacency_matrix
from rw_simulate import initial_rw_decomposition, simulate_graph, simulate_circuit
from tree import incidence_list, calc_partitions, calc_ranks


def gen_graph(n, m):
    g = zx.Graph()
    for _ in range(n):
        g.add_vertex(zx.VertexType.Z, phase=random.randint(0, 7) / 4)
    while g.num_edges() < m:
        [u, v] = random.sample(range(n), k=2)
        if (u, v) not in g.edge_set() and (v, u) not in g.edge_set():
            g.add_edge((u, v), zx.EdgeType.HADAMARD)
    return g


def check_answer(res, corr, g):
    if not np.allclose(res, corr):
        print(f'WA {res} {corr}')
        print(g)
        print(g.edge_set())
        assert False


def check_circuit_simulation(circ, state, effect):
    res = simulate_circuit(circ, state, effect)
    g = circ.to_graph()
    g.apply_state(state)
    g.apply_effect(effect)
    zx.simplify.full_reduce(g)
    corr = zx.tensorfy(g)
    check_answer(res, corr, g)


def test_rank_factorize():
    A = GF2([[1, 1, 1, 1, 1],
             [1, 1, 1, 1, 1],
             [0, 1, 0, 0, 1],
             [1, 0, 1, 1, 0]])
    r, U, V = rank_factorize(A)
    assert r == 2
    assert (U[:, :r] @ V[:r] == A).all()


def test_generalized_inverse():
    A = GF2([[1, 1, 1, 0],
             [1, 1, 1, 1],
             [1, 1, 1, 1]])
    B = generalized_inverse(A)
    assert (A @ B @ A == A).all()


def test_initial_rw_decomposition():
    n_qubits, n_gates_start = 10, 50
    for it in range(10):
        n_gates = n_gates_start * (it + 1)
        circ = zx.generate.CNOT_HAD_PHASE_circuit(qubits=n_qubits, depth=n_gates)
        g = circ.to_graph()
        zx.full_reduce(g)
        tree_edges = initial_rw_decomposition(g)
        inc = incidence_list(tree_edges)
        part = calc_partitions(inc, tree_edges)
        mat, _, _ = adjacency_matrix(g)
        tree_edges = calc_ranks(mat, part, tree_edges)
        r = max(r for _, _, r in tree_edges)
        assert r <= n_qubits


def test_rw_simulate_one_edge():
    g = zx.Graph()
    g.add_vertex(zx.VertexType.Z, phase=0)
    g.add_vertex(zx.VertexType.Z, phase=1)
    g.add_edge((0, 1), zx.EdgeType.HADAMARD)
    tree_edges = [(0, 1), (1, 0)]
    res = simulate_graph(g, tree_edges)
    assert np.allclose(res, zx.tensorfy(g))


def test_rw_simulate_square():
    g = zx.Graph()
    for i in range(4):
        g.add_vertex(zx.VertexType.Z)
    for i in range(4):
        g.add_edge((i, (i + 1) % 4), zx.EdgeType.HADAMARD)
    tree_edges = [(0, 4), (4, 0), (1, 4), (4, 1), (2, 5), (5, 2), (3, 5), (5, 3), (4, 5), (5, 4)]
    res = simulate_graph(g, tree_edges)
    assert np.allclose(res, zx.tensorfy(g))


def test_rw_simulate_random_graph():
    n, m = 10, 40
    for _ in range(10):
        g = gen_graph(n, m)
        res = simulate_graph(g)
        corr = zx.tensorfy(g)
        check_answer(res, corr, g)


def test_rw_simulate_random_circuit():
    n_qubits, n_gates = 5, 100
    basic_states = ['0', '1', '+', '-']
    for _ in range(10):
        check_circuit_simulation(
            zx.generate.CNOT_HAD_PHASE_circuit(qubits=n_qubits, depth=n_gates),
            random.choice(basic_states) * n_qubits,
            random.choice(basic_states) * n_qubits
        )


def test_rw_simulate_small_circuit():
    check_circuit_simulation(zx.generate.CNOT_HAD_PHASE_circuit(qubits=1, depth=0),
                             '0', '0')
    check_circuit_simulation(zx.generate.CNOT_HAD_PHASE_circuit(qubits=1, depth=1, p_t=0.8),
                             '+', '+')
    check_circuit_simulation(zx.generate.CNOT_HAD_PHASE_circuit(qubits=2, depth=0),
                             '1-', '-1')
    check_circuit_simulation(zx.generate.CNOT_HAD_PHASE_circuit(qubits=2, depth=1),
                             '--', '11')
