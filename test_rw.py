import numpy as np
import pyzx as zx
import random

from gf2_factorize import gf2_factorize
from rank_width import rw_decomposition
from simulate_rw import simulate_rw
from tree import incidence_list, calc_partitions


def gen_graph(n, m):
    g = zx.Graph()
    for _ in range(n):
        g.add_vertex(zx.VertexType.Z)
    while g.num_edges() < m:
        [u, v] = random.sample(range(n), k=2)
        g.add_edge((u, v))
    return g


def test_rank_decomposition():
    n, m = 10, 20
    g = gen_graph(n, m)
    tree_edges = rw_decomposition(g)
    inc = incidence_list(tree_edges)
    part = calc_partitions(inc, tree_edges)
    mat = np.zeros((n, n), dtype=np.int32)
    for u, v in g.edge_set():
        mat[u][v] = mat[v][u] = 1
    for e, (_, _, r) in enumerate(tree_edges):
        A = mat[part[e]][:, ~part[e]]
        U, V = gf2_factorize(A)
        assert (U @ V % 2 == A).all()
        assert U.shape[1] == r


def test_rw_simulate():
    n, m = 10, 20
    g = gen_graph(n, m)
    tree_edges = rw_decomposition(g)
    simulate_rw(g, tree_edges)
