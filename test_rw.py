import numpy as np
import pyzx as zx
import random
from galois import GF2

from gf2 import rank_factorize, generalized_inverse
from rank_width import rw_decomposition
from simulate_rw import simulate_rw
from tree import incidence_list, calc_partitions


def gen_graph(n, m):
    g = zx.Graph()
    for _ in range(n):
        g.add_vertex(zx.VertexType.Z, phase=random.randint(0, 7) / 4)
    while g.num_edges() < m:
        [u, v] = random.sample(range(n), k=2)
        if (u, v) not in g.edge_set() and (v, u) not in g.edge_set():
            g.add_edge((u, v), zx.EdgeType.HADAMARD)
    return g


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


def test_rw_decomposition():
    n, m = 10, 20
    g = gen_graph(n, m)
    tree_edges = rw_decomposition(g)
    inc = incidence_list(tree_edges)
    part = calc_partitions(inc, tree_edges)
    mat = GF2.Zeros((n, n))
    for u, v in g.edge_set():
        mat[u][v] = mat[v][u] = 1
    for e, (_, _, rw) in enumerate(tree_edges):
        A = mat[part[e]][:, ~part[e]]
        r, U, V = rank_factorize(A)
        assert (U[:, :r] @ V[:r] == A).all()
        assert r == rw


def test_rw_simulate_one_edge():
    g = zx.Graph()
    g.add_vertex(zx.VertexType.Z, phase=0)
    g.add_vertex(zx.VertexType.Z, phase=1)
    g.add_edge((0, 1), zx.EdgeType.HADAMARD)
    tree_edges = [(0, 1, 1), (1, 0, 1)]
    res = simulate_rw(g, tree_edges)
    assert np.allclose(res, zx.tensorfy(g))


def test_rw_simulate_square():
    g = zx.Graph()
    for i in range(4):
        g.add_vertex(zx.VertexType.Z)
    for i in range(4):
        g.add_edge((i, (i + 1) % 4), zx.EdgeType.HADAMARD)
    tree_edges = [(0, 4, 1), (4, 0, 1), (1, 4, 1), (4, 1, 1),
                  (2, 5, 1), (5, 2, 1), (3, 5, 1), (5, 3, 1), (4, 5, 2), (5, 4, 2)]
    res = simulate_rw(g, tree_edges)
    assert np.allclose(res, zx.tensorfy(g))


def test_rw_simulate():
    n, m = 10, 40
    g = gen_graph(n, m)
    tree_edges = rw_decomposition(g)
    res = simulate_rw(g, tree_edges)
    corr = zx.tensorfy(g, preserve_scalar=False)
    assert np.allclose(res, corr)
