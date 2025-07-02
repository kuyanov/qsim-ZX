import numpy as np
import pyzx as zx

from gf2_factorize import gf2_factorize
from tree import incidence_list, calc_partitions


def simulate_rw(g: zx.graph.base.BaseGraph, tree_edges):
    inc = incidence_list(tree_edges)
    part = calc_partitions(inc, tree_edges)
    n = g.num_vertices()

    def dfs(e):
        u, _, r = tree_edges[e]
        if len(inc[u]) == 1:
            phase = g.phase(u)
            state = np.array([(1 + np.exp(1j * phase)) / 2 ** 0.5,
                              (1 - np.exp(1j * phase)) / 2 ** 0.5])
            nb = list(g.neighbors(u))
            conn = np.zeros((1, n), dtype=np.int32)
            conn[0][nb] = 1
            return state, conn
        children = [f for f in inc[u] if e ^ f != 1]
        assert len(children) == 2
        f1, f2 = children
        v, _, r_v = tree_edges[f1]
        w, _, r_w = tree_edges[f2]
        state_v, conn_v = dfs(f1)
        state_w, conn_w = dfs(f2)
        conn_full = np.concatenate((conn_v, conn_w))
        conn_in = conn_full[:, part[e]].copy()
        conn_full[:, part[e]] = 0
        fac_l, fac_r = gf2_factorize(conn_full)
        assert fac_l.shape[1] == r
        return np.zeros([2] * r), fac_r

    _, e = max([(r, e) for e, (_, _, r) in enumerate(tree_edges)])
    dfs(e)
    dfs(e ^ 1)
