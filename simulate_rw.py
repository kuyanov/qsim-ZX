import numpy as np
import pyzx as zx
from galois import GF2
from itertools import product
from math import *

from gf2 import rank_factorize, generalized_inverse
from tree import incidence_list, calc_partitions


def simulate_rw(g: zx.graph.base.BaseGraph, tree_edges):
    inc = incidence_list(tree_edges)
    part = calc_partitions(inc, tree_edges)
    n = g.num_vertices()
    mat = GF2.Zeros((n, n))
    for u, v in g.edge_set():
        mat[u][v] = mat[v][u] = 1

    def conv_naive(S, T, A, B):
        r_u, r_v, r_w = A.shape[1], len(S.shape), len(T.shape)
        A1, A2 = A[:r_v], A[r_v:]
        R = np.zeros([2] * r_u, dtype=S.dtype)
        for x in product(range(2), repeat=r_u):
            for a in product(range(2), repeat=r_v):
                for b in product(range(2), repeat=r_w):
                    x1, a1, b1 = GF2(x), GF2(a), GF2(b)
                    phase = int(np.dot(a1, A1 @ x1) + np.dot(b1, A2 @ x1) + np.dot(a1, B @ b1))
                    R[x] += S[a] * T[b] * (-1) ** phase
        return R / sqrt(2) ** ((A == 1).sum() + (B == 1).sum())

    def conv_final(S, T, B):
        r_v, r_w = len(S.shape), len(T.shape)
        res = 0
        for a in product(range(2), repeat=r_v):
            for b in product(range(2), repeat=r_w):
                a1, b1 = GF2(a), GF2(b)
                phase = int(np.dot(a1, B @ b1))
                res += S[a] * T[b] * (-1) ** phase
        return res / sqrt(2) ** (B == 1).sum()

    def dfs(e):
        u, _, r_u = tree_edges[e]
        if len(inc[u]) == 1:
            phase = g.phase(u)
            state = np.array([1, np.exp(pi * 1j * phase)])
            nb = list(g.neighbors(u))
            conn = GF2.Zeros((1, n))
            conn[0][nb] = 1
            return state, conn

        children = [f for f in inc[u] if e ^ f != 1]
        assert len(children) == 2
        f1, f2 = children
        v, _, r_v = tree_edges[f1]
        w, _, r_w = tree_edges[f2]
        state_v, conn_v = dfs(f1)
        state_w, conn_w = dfs(f2)
        conn_vw = conn_v[:, part[f2]]
        conn_wv = conn_w[:, part[f1]]
        conn_out = GF2(np.concatenate((conn_v, conn_w)))
        conn_out[:, part[e]] = 0

        r, fac_l, fac_r = rank_factorize(conn_out)
        fac_l, fac_r = fac_l[:, :r], fac_r[:r]
        assert r == r_u

        mat_in = mat[part[f1]][:, part[f2]]
        mat_in_inv = generalized_inverse(mat_in)
        conn_in = conn_vw @ mat_in_inv @ conn_wv.T
        state_u_hat = conv_naive(state_v, state_w, fac_l, conn_in)
        state_u = np.fft.fftn(state_u_hat) / sqrt(2) ** r_u

        sc_cnt = (conn_out == 1).sum() - (fac_l == 1).sum() - (fac_r == 1).sum() + r_u
        sc_cnt += -(mat_in == 1).sum()
        sc_cnt += (conn_vw == 1).sum() + (conn_wv == 1).sum() - (conn_in == 1).sum()
        return state_u / sqrt(2) ** sc_cnt, fac_r

    r, e = max([(r, e) for e, (_, _, r) in enumerate(tree_edges)])
    state_l, conn_l = dfs(e)
    state_r, conn_r = dfs(e ^ 1)
    conn_lr = conn_l[:, part[e ^ 1]]
    conn_rl = conn_r[:, part[e]]
    mat_in = mat[part[e]][:, part[e ^ 1]]
    mat_in_inv = generalized_inverse(mat_in)
    conn_in = conn_lr @ mat_in_inv @ conn_rl.T
    res = conv_final(state_l, state_r, conn_in)
    sc_cnt = (conn_lr == 1).sum() + (conn_rl == 1).sum() - (conn_in == 1).sum() - (mat_in == 1).sum()
    return res / sqrt(2) ** sc_cnt
