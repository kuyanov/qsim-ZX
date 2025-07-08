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

    def count(A: GF2):
        return (A == 1).sum()

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
        return R / sqrt(2) ** (count(A) + count(B))

    def conv_ruv(S, T, A, B):
        r_u, r_v, r_w = A.shape[1], len(S.shape), len(T.shape)
        A1, A2 = A[:r_v], A[r_v:]
        if r_w > r_v:
            S, T = T, S
            r_v, r_w = r_w, r_v
            A1, A2 = A2, A1
            B = B.T
        S_hat = np.fft.fftn(S)
        R = np.zeros([2] * r_u, dtype=S.dtype)
        for x in product(range(2), repeat=r_u):
            for b in product(range(2), repeat=r_w):
                x1, b1 = GF2(x), GF2(b)
                a = tuple(A1 @ x1 + B @ b1)
                phase = int(np.dot(b1, A2 @ x1))
                R[x] += T[b] * S_hat[a] * (-1) ** phase
        return R / sqrt(2) ** (count(A) + count(B))

    def merge_children(f1, f2):
        v, _, r_v = tree_edges[f1]
        w, _, r_w = tree_edges[f2]
        state_v, conn_v = dfs(f1)
        state_w, conn_w = dfs(f2)

        conn_vw = conn_v[:, part[f2]]
        conn_wv = conn_w[:, part[f1]]
        conn_out = GF2(np.concatenate((conn_v, conn_w)))
        conn_out[:, part[f1] | part[f2]] = 0

        r_u, fac_l, fac_r = rank_factorize(conn_out)
        fac_l, fac_r = fac_l[:, :r_u], fac_r[:r_u]

        mat_in = mat[part[f1]][:, part[f2]]
        mat_inv = generalized_inverse(mat_in)
        conn_in = conn_vw @ mat_inv @ conn_wv.T
        state_u_hat = conv_ruv(state_v, state_w, fac_l, conn_in)
        state_u = np.fft.fftn(state_u_hat) / sqrt(2) ** r_u

        sc_cnt = count(conn_out) - count(fac_l) - count(fac_r) + r_u
        sc_cnt += count(conn_vw) + count(conn_wv) - count(conn_in) - count(mat_in)
        return state_u / sqrt(2) ** sc_cnt, fac_r

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
        return merge_children(*children)

    r, e = max([(r, e) for e, (_, _, r) in enumerate(tree_edges)])
    return merge_children(e, e ^ 1)[0]
