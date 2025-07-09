import numpy as np
import pyzx as zx
from galois import GF2
from itertools import product
from math import *

from gf2 import rank_factorize, generalized_inverse
from tree import incidence_list, calc_partitions


def count1(A: GF2):
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
    return R / sqrt(2) ** (count1(A) + count1(B))


def conv_ruv(S, T, A1, A2, B):
    r_u, r_v, r_w = A1.shape[1], len(S.shape), len(T.shape)
    S_hat = np.fft.fftn(S)
    R = np.zeros([2] * r_u, dtype=S.dtype)
    for x in product(range(2), repeat=r_u):
        for b in product(range(2), repeat=r_w):
            x1, b1 = GF2(x), GF2(b)
            a = tuple(A1 @ x1 + B @ b1)
            phase = int(np.dot(b1, A2 @ x1))
            R[x] += T[b] * S_hat[a] * (-1) ** phase
    return R / sqrt(2) ** (count1(A1) + count1(A2) + count1(B))


def conv_rvw(S, T, A1, A2, B):
    r_u, r_v, r_w = A1.shape[1], len(S.shape), len(T.shape)
    F = np.zeros(S.shape + T.shape, dtype=S.dtype)
    for a in product(range(2), repeat=r_v):
        for b in product(range(2), repeat=r_w):
            a1, b1 = GF2(a), GF2(b)
            F[*a, *b] = S[a] * T[b] * (-1) ** int(np.dot(a1, B @ b1))
    F_hat = np.fft.fftn(F)
    R = np.zeros([2] * r_u, dtype=S.dtype)
    for x in product(range(2), repeat=r_u):
        x1 = GF2(x)
        R[x] = F_hat[*(A1 @ x1), *(A2 @ x1)]
    return R / sqrt(2) ** (count1(A1) + count1(A2) + count1(B))


def conv(S, T, A, B):
    r_u, r_v, r_w = A.shape[1], len(S.shape), len(T.shape)
    r_max = max(r_u, r_v, r_w)
    if r_u == r_max:
        return conv_rvw(S, T, A[:r_v], A[r_v:], B)
    elif r_v == r_max:
        return conv_ruv(T, S, A[r_v:], A[:r_v], B.T)
    else:
        return conv_ruv(S, T, A[:r_v], A[r_v:], B)


def simulate_rw(g: zx.graph.base.BaseGraph, tree_edges, preserve_scalar=True):
    inc = incidence_list(tree_edges)
    part = calc_partitions(inc, tree_edges)
    id2vert = list(g.vertices())
    vert2id = {v: i for i, v in enumerate(g.vertices())}
    n = g.num_vertices()
    mat = GF2.Zeros((n, n))
    for u, v in g.edge_set():
        mat[vert2id[u]][vert2id[v]] = mat[vert2id[v]][vert2id[u]] = 1

    def merge_children(f1, f2):
        v, _, r_v = tree_edges[f1]
        w, _, r_w = tree_edges[f2]
        state_v, conn_v = simplify_recursive(f1)
        state_w, conn_w = simplify_recursive(f2)

        conn_vw = conn_v[:, part[f2]]
        conn_wv = conn_w[:, part[f1]]
        conn_out = GF2(np.concatenate((conn_v, conn_w)))
        conn_out[:, part[f1] | part[f2]] = 0

        r_u, fac_l, fac_r = rank_factorize(conn_out)
        fac_l, fac_r = fac_l[:, :r_u], fac_r[:r_u]

        mat_in = mat[part[f1]][:, part[f2]]
        mat_inv = generalized_inverse(mat_in)
        conn_in = conn_vw @ mat_inv @ conn_wv.T
        state_u_hat = conv(state_v, state_w, fac_l, conn_in)
        state_u = np.fft.fftn(state_u_hat) / sqrt(2) ** r_u

        sc_cnt = count1(conn_out) - count1(fac_l) - count1(fac_r) + r_u
        sc_cnt += count1(conn_vw) + count1(conn_wv) - count1(conn_in) - count1(mat_in)
        return state_u / sqrt(2) ** sc_cnt, fac_r

    def simplify_recursive(e):
        u, _, r_u = tree_edges[e]
        if len(inc[u]) == 1:
            phase = g.phase(id2vert[u])
            state = np.array([1, np.exp(pi * 1j * phase)])
            nb = [vert2id[v] for v in g.neighbors(id2vert[u])]
            conn = GF2.Zeros((1, n))
            conn[0][nb] = 1
            return state, conn

        children = [f for f in inc[u] if e ^ f != 1]
        return merge_children(*children)

    if not tree_edges:
        return g.scalar.to_number() ** preserve_scalar
    r, e = max([(r, e) for e, (_, _, r) in enumerate(tree_edges)])
    return merge_children(e, e ^ 1)[0] * g.scalar.to_number() ** preserve_scalar
