import numpy as np
import pyzx as zx
import quizx
from galois import GF2
from itertools import product
from math import *

from decomposition import pauli_flow_decomposition, sub_decomposition
from gf2 import rank_factorize, generalized_inverse
from graph import adjacency_matrix


def weight(A: GF2) -> int:
    return int((A == 1).sum())


def bitmasks(r):
    return (np.arange(2 ** r)[None, :] & (1 << np.arange(r))[:, None] > 0).astype(np.uint32)


def conv_naive(S: np.ndarray, T: np.ndarray, A: GF2, B: GF2) -> np.ndarray:
    r_u, r_v, r_w = A.shape[1], len(S.shape), len(T.shape)
    A1, A2 = A[:r_v], A[r_v:]
    R = np.zeros([2] * r_u, dtype=S.dtype)
    for x in product(range(2), repeat=r_u):
        for a in product(range(2), repeat=r_v):
            for b in product(range(2), repeat=r_w):
                x1, a1, b1 = GF2(x), GF2(a), GF2(b)
                phase = int(np.dot(a1, A1 @ x1) + np.dot(b1, A2 @ x1) + np.dot(a1, B @ b1))
                R[x] += S[a] * T[b] * (-1) ** phase
    return R / sqrt(2) ** (weight(A) + weight(B))


def conv_ruw(S: np.ndarray, T: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    r_u, r_v, r_w = A.shape[1], B.shape[0], B.shape[1]
    S_hat = np.fft.fftn(S)
    xk, bj = bitmasks(r_u), bitmasks(r_w)
    Axk = A @ xk % 2
    Bbj = B @ bj % 2
    R = np.zeros((2,) * r_u, dtype=np.complex128)
    for ix, x in enumerate(xk.T):
        ai = (Axk[:r_v, [ix]] + Bbj) % 2
        R[tuple(x)] = (S_hat[tuple(ai)] * T[tuple(bj)] * (-1) ** (Axk[r_v:, ix].T @ bj).astype(int)).sum()
    return R / sqrt(2) ** (A.sum() + B.sum())


def conv_rvw(S: np.ndarray, T: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    r_u, r_v, r_w = A.shape[1], B.shape[0], B.shape[1]
    xk, ai, bj = bitmasks(r_u), bitmasks(r_v), bitmasks(r_w)
    phase = (ai.T @ B @ bj).reshape((2,) * (r_v + r_w), order='F').astype(int)
    F = np.tensordot(S, T, axes=0) * (-1) ** phase
    F_hat = np.fft.fftn(F)
    Axk = A @ xk % 2
    R = F_hat[tuple(Axk)].reshape((2,) * r_u, order='F')
    return R / sqrt(2) ** (A.sum() + B.sum())


def conv(S: np.ndarray, T: np.ndarray, A: GF2, B: GF2) -> np.ndarray:
    r_u, r_v, r_w = A.shape[1], B.shape[0], B.shape[1]
    r_max = max(r_u, r_v, r_w)
    A = np.array(A, dtype=np.uint32)
    B = np.array(B, dtype=np.uint32)
    if r_u == r_max:
        return conv_rvw(S, T, A, B)
    elif r_v == r_max:
        return conv_ruw(S, T, A, B)
    else:
        return conv_ruw(T, S, np.concatenate([A[r_v:], A[:r_v]]), B.T)


def simulate_graph(g: zx.graph.base.BaseGraph, decomp, preserve_scalar=True) -> complex:
    n = g.num_vertices()
    v_all = np.array(list(g.vertices()))
    v_id = {v: i for i, v in enumerate(v_all)}

    def iterate(elem):
        if isinstance(elem, int):
            phase = g.phase(elem)
            state = np.array([1, np.exp(pi * 1j * phase)])
            conn = adjacency_matrix(g, [elem], v_all)
            used = np.zeros(n, dtype=bool)
            used[v_id[elem]] = True
            return used, state, conn

        used_v, state_v, conn_v = iterate(elem[0])
        used_w, state_w, conn_w = iterate(elem[1])
        conn_vw = conn_v[:, used_w]
        conn_wv = conn_w[:, used_v]
        conn_out = GF2(np.concatenate((conn_v, conn_w)))
        conn_out[:, used_v | used_w] = 0

        r_u, fac_l, fac_r = rank_factorize(conn_out)
        fac_l, fac_r = fac_l[:, :r_u], fac_r[:r_u]

        mat_in = adjacency_matrix(g, v_all[used_v], v_all[used_w])
        mat_inv = generalized_inverse(mat_in)
        conn_in = conn_vw @ mat_inv @ conn_wv.T
        state_u_hat = conv(state_v, state_w, fac_l, conn_in)
        state_u = np.fft.fftn(state_u_hat) / sqrt(2) ** r_u
        pw = weight(conn_out) - weight(fac_l) - weight(fac_r) + r_u
        pw += weight(conn_vw) + weight(conn_wv) - weight(conn_in) - weight(mat_in)
        return used_v | used_w, state_u / sqrt(2) ** pw, fac_r

    if n == 0:
        return g.scalar.to_number() ** preserve_scalar
    return iterate(decomp)[1].item() * g.scalar.to_number() ** preserve_scalar


def simulate_circuit(circ: zx.Circuit, state: str, effect: str) -> complex:
    g = circ.to_graph()
    zx.full_reduce(g)
    decomp = pauli_flow_decomposition(g)
    g.apply_state(state)
    g.apply_effect(effect)
    zx.clifford_simp(g)
    if g.num_vertices() == 0:
        return simulate_graph(g, [])
    g2 = g.copy(backend="quizx-vec")
    decomp = sub_decomposition(decomp, list(g.vertices()), list(g2.vertices()))
    init_decomp = quizx.DecompTree.from_list(decomp)
    ann = quizx.RankwidthAnnealer(g2, init_decomp=init_decomp)
    final_decomp = ann.run()
    return simulate_graph(g2, final_decomp.to_list())
