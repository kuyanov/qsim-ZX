import numpy as np
import pyzx as zx
from galois import GF2

from gf2 import rank_factorize


def adjacency_matrix(g: zx.graph.base.BaseGraph) -> (GF2, dict, list):
    id2vert = list(g.vertices())
    vert2id = {v: i for i, v in enumerate(g.vertices())}
    n = g.num_vertices()
    mat = GF2.Zeros((n, n))
    for u, v in g.edge_set():
        mat[vert2id[u]][vert2id[v]] = mat[vert2id[v]][vert2id[u]] = 1
    return mat, vert2id, id2vert


def cut_rank(mat: GF2, used: np.array) -> int:
    A = mat[used][:, ~used]
    r, _, _ = rank_factorize(A)
    return r
