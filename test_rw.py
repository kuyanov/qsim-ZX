import numpy as np
import pyzx as zx
import random

from gf2_algebra import *
from rank_width import rank_decomposition
from simulate_rw import simulate_rw
from tree import *


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
    rw, edges = rank_decomposition(g)
    inc = incidence_list(edges)
    root = len(inc) - 1
    children = root_tree(inc, root)
    tour, leaf_seg = euler_tour(children, root)
    mat = np.zeros((n, n), dtype=np.int32)
    for u, v in g.edge_set():
        mat[u][v] = mat[v][u] = 1
    for u in range(len(children)):
        for v, r in children[u]:
            vs1, vs2 = get_partition(tour, leaf_seg, v)
            A = mat[vs1][:, vs2]
            U, V = gf2_factorize(A)
            assert (U @ V % 2 == A).all()
            assert U.shape[1] == r


def test_rw_simulate():
    n, m = 10, 20
    g = gen_graph(n, m)
    r, rw_edges = rank_decomposition(g)
    simulate_rw(g, rw_edges)
