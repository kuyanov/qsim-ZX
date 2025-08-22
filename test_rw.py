import numpy as np
import pyzx as zx
import quizx
import random
from fractions import Fraction
from galois import GF2

from decomposition import pauli_flow_decomposition, rank_width
from gf2 import rank_factorize, generalized_inverse
from rw_simulate import conv_naive, conv_vw, conv_uv, conv_uw, simulate_graph, simulate_circuit


def generate_graph(n, m):
    g = zx.Graph()
    for _ in range(n):
        g.add_vertex(zx.VertexType.Z, phase=Fraction(random.randint(0, 7), 4))
    while g.num_edges() < m:
        [u, v] = random.sample(range(n), k=2)
        if (u, v) not in g.edge_set() and (v, u) not in g.edge_set():
            g.add_edge((u, v), zx.EdgeType.HADAMARD)
    return g


def check_amplitude(res, corr, info=''):
    if not np.allclose(res, corr):
        print(f'Amplitude mismatch: expected {corr}, found {res}')
        print(info)
        assert False


def check_graph_simulation(g, decomp):
    res = simulate_graph(g, decomp)
    corr = zx.tensorfy(g)
    check_amplitude(res, corr, str(g.edge_set()))


def check_circuit_simulation(circ, state, effect):
    res = simulate_circuit(circ, state, effect)
    g = circ.to_graph()
    g.apply_state(state)
    g.apply_effect(effect)
    corr = zx.tensorfy(g)
    zx.simplify.full_reduce(g)
    check_amplitude(res, corr, str(g.edge_set()))


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


def test_pauli_flow_decomposition():
    n_qubits, n_gates_start = 10, 50
    for it in range(5):
        n_gates = n_gates_start * (it + 1)
        circ = zx.generate.CNOT_HAD_PHASE_circuit(qubits=n_qubits, depth=n_gates)
        g = circ.to_graph()
        zx.full_reduce(g)
        decomp = pauli_flow_decomposition(g)
        assert rank_width(decomp, g) <= n_qubits


def test_quizx_annealer():
    for _ in range(50):
        g = generate_graph(10, 40).copy(backend="quizx-vec")
        ann = quizx.RankwidthAnnealer(g)
        decomp = ann.run()
        assert decomp.rankwidth(g) == rank_width(decomp.to_list(), g)


def test_convolution():
    r_u, r_v, r_w = 2, 2, 2
    Psi_v = np.random.random((2,) * r_v).astype(np.complex128)
    Psi_w = np.random.random((2,) * r_w).astype(np.complex128)
    E_vw = np.random.randint(2, size=(r_v, r_w))
    E_vu = np.random.randint(2, size=(r_v, r_u))
    E_wu = np.random.randint(2, size=(r_w, r_u))
    corr = conv_naive(Psi_v, Psi_w, E_vw, E_vu, E_wu)
    res_vw = conv_vw(Psi_v, Psi_w, E_vw, E_vu, E_wu)
    res_uv = conv_uv(Psi_v, Psi_w, E_vw, E_vu, E_wu)
    res_uw = conv_uw(Psi_v, Psi_w, E_vw, E_vu, E_wu)
    check_amplitude(res_vw, corr, 'conv_vw')
    check_amplitude(res_uv, corr, 'conv_uv')
    check_amplitude(res_uw, corr, 'conv_uw')


def test_simulate_one_edge():
    g = zx.Graph()
    g.add_vertex(zx.VertexType.Z, phase=0)
    g.add_vertex(zx.VertexType.Z, phase=1)
    g.add_edge((0, 1), zx.EdgeType.HADAMARD)
    decomp = [0, 1]
    check_graph_simulation(g, decomp)


def test_simulate_square():
    g = zx.Graph()
    for i in range(4):
        g.add_vertex(zx.VertexType.Z)
    for i in range(4):
        g.add_edge((i, (i + 1) % 4), zx.EdgeType.HADAMARD)
    decomp = [[0, 1], [2, 3]]
    check_graph_simulation(g, decomp)


def test_simulate_random_graph():
    for _ in range(10):
        g = generate_graph(10, 40).copy(backend="quizx-vec")
        ann = quizx.RankwidthAnnealer(g)
        check_graph_simulation(g, ann.init_decomp().to_list())


def test_simulate_random_circuit():
    n_qubits, n_gates = 10, 200
    basic_states = ['0', '1', '+', '-']
    for _ in range(10):
        check_circuit_simulation(
            zx.generate.CNOT_HAD_PHASE_circuit(qubits=n_qubits, depth=n_gates),
            random.choice(basic_states) * n_qubits,
            random.choice(basic_states) * n_qubits
        )


def test_simulate_small_circuit():
    check_circuit_simulation(zx.generate.CNOT_HAD_PHASE_circuit(qubits=1, depth=0),
                             '0', '0')
    check_circuit_simulation(zx.generate.CNOT_HAD_PHASE_circuit(qubits=1, depth=1, p_t=0.8),
                             '+', '+')
    check_circuit_simulation(zx.generate.CNOT_HAD_PHASE_circuit(qubits=2, depth=0),
                             '1-', '-1')
    check_circuit_simulation(zx.generate.CNOT_HAD_PHASE_circuit(qubits=2, depth=1),
                             '--', '11')
