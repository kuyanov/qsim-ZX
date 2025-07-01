import numpy as np

from gf2_algebra import *
from tree import *


def simulate_rw(g, rw_edges):
    inc = incidence_list(rw_edges)
    n = len(inc)
    # root = len(inc) - 1
    # children = root_tree(inc, root)
    # tour, leaf_seg = euler_tour(children, root)
    # n = g.num_vertices()
    # mat = np.zeros((n, n), dtype=np.bool_)

    def dfs(u, p, r):
        if len(inc[u]) == 1:
            phase = g.phase(u)
            state = np.array([(1 + np.exp(1j * phase)) / 2 ** 0.5,
                              (1 - np.exp(1j * phase)) / 2 ** 0.5])
            nb = list(g.neighbors(u))
            mat = np.zeros((1, n), dtype=np.int32)
            mat[0][nb] = True
            used = np.zeros(n, dtype=np.bool_)
            used[u] = True
            return used, state, mat
        children = [e for e in inc[u] if e != (p, r)]
        assert len(children) == 2
        v, r_v = children[0]
        w, r_w = children[1]
        used_v, state_v, mat_v = dfs(v, u, r_v)
        used_w, state_w, mat_w = dfs(w, u, r_w)
        used = used_v | used_w
        mat_full = np.concatenate((mat_v, mat_w))
        mat_in = mat_full[:,used].copy()
        mat_full[:,used] = False
        fac_l, fac_r = gf2_factorize(mat_full)
        assert fac_l.shape[1] == r
        return used, np.zeros([2] * r), fac_r

    rs = [r for u, v, r in rw_edges]
    ind_max = rs.index(max(rs))
    u, v, r_max = rw_edges[ind_max]
    dfs(u, v, r_max)
    dfs(v, u, r_max)
