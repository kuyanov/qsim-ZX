import pyzx as zx
from galois import GF2
from typing import Iterable


def adjacency_matrix(g: zx.graph.base.BaseGraph, v_left: Iterable[int], v_right: Iterable[int]) -> GF2:
    left_id = {u: i for i, u in enumerate(v_left)}
    right_id = {v: i for i, v in enumerate(v_right)}
    n, m = len(left_id), len(right_id)
    mat = GF2.Zeros((n, m))
    for u, v in g.edge_set():
        if u in left_id and v in right_id:
            mat[left_id[u]][right_id[v]] = 1
        if v in left_id and u in right_id:
            mat[left_id[v]][right_id[u]] = 1
    return mat
