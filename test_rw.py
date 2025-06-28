import numpy as np
import pyzx as zx
import random

from gf2_algebra import gf2_rank
from rank_width import rank_decomposition
from simulate_rw import simulate_rw
from tree import *


def test_rank_decomposition():
    n, m = 10, 20
    g = zx.Graph()
    for _ in range(n):
        g.add_vertex(zx.VertexType.Z)
    while g.num_edges() < m:
        [u, v] = random.sample(range(n), k=2)
        g.add_edge((u, v))
    rw, edges = rank_decomposition(g)
    inc = incidence_list(edges)
    root = len(inc) - 1
    children = root_tree(inc, root)
    tour, leaf_seg = euler_tour(children, root)
    mat = np.zeros((n, n), dtype=np.bool_)
    for u, v in g.edge_set():
        mat[u][v] = mat[v][u] = True
    for u in range(len(children)):
        for v, r in children[u]:
            vs1, vs2 = get_partition(tour, leaf_seg, v)
            assert gf2_rank(mat[vs1][:, vs2]) == r
