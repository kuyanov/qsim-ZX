import numpy as np
import pyzx as zx
from galois import GF2
from itertools import product
from math import *
from pyzx.gflow import gflow

from gf2 import rank_factorize, generalized_inverse
from graph import adjacency_matrix
from rank_width import rw_decomposition
from tree import incidence_list, calc_partitions, calc_ranks, induced_tree


def count1(A: GF2) -> int:
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
    return R / sqrt(2) ** (count1(A) + count1(B))


def conv_ruw(S: np.ndarray, T: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    r_u, r_v, r_w = A.shape[1], B.shape[0], B.shape[1]
    S_hat = np.fft.fftn(S)
    xk, bj = bitmasks(r_u), bitmasks(r_w)
    Axk = A @ xk % 2
    Bbj = B @ bj % 2
    R = np.zeros((2,) * r_u, dtype=np.complex_)
    for ix, x in enumerate(xk.T):
        ai = (Axk[:r_v, [ix]] + Bbj) % 2
        R[tuple(x)] = (S_hat[tuple(ai)] * T[tuple(bj)] * (-1) ** (Axk[r_v:, ix].T @ bj)).sum()
    return R / sqrt(2) ** (A.sum() + B.sum())


def conv_rvw(S: np.ndarray, T: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    r_u, r_v, r_w = A.shape[1], B.shape[0], B.shape[1]
    xk, ai, bj = bitmasks(r_u), bitmasks(r_v), bitmasks(r_w)
    phase = (ai.T @ B @ bj).reshape((2,) * (r_v + r_w), order='F')
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


def simulate_graph(g: zx.graph.base.BaseGraph, tree_edges=None, preserve_scalar=True) -> complex:
    if tree_edges is None:
        tree_edges = rw_decomposition(g)
    n = g.num_vertices()
    inc = incidence_list(tree_edges)
    part = calc_partitions(inc, tree_edges)
    mat, vert2id, id2vert = adjacency_matrix(g)

    def merge_children(f1, f2):
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
        u = tree_edges[e][0]
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
    tree_edges = calc_ranks(mat, part, tree_edges)
    r, e = max([(r, e) for e, (_, _, r) in enumerate(tree_edges)])
    return merge_children(e, e ^ 1)[0] * g.scalar.to_number() ** preserve_scalar


def extract_phase_gadgets(g: zx.graph.base.BaseGraph) -> list:
    phase_gadgets = []
    for v in g.vertices():
        if g.vertex_degree(v) != 1:
            continue
        u = list(g.neighbors(v))[0]
        if v in g.inputs() or v in g.outputs() or g.phase(u) != 0:
            continue
        phase_gadgets.append((v, u))
    return phase_gadgets


def initial_rw_decomposition(g: zx.graph.base.BaseGraph) -> list:
    layers, _ = gflow(g, pauli=True)
    phase_gadgets = extract_phase_gadgets(g)
    gadget_dict = dict()
    for i, (u, v) in enumerate(phase_gadgets):
        gadget_dict[u] = gadget_dict[v] = (u, v)
    ord = (list(g.inputs()) +
           sorted(list(layers.keys()), key=lambda v: layers[v]) +
           list(g.outputs()))
    n = g.num_vertices()
    assert len(ord) == n
    mat, vert2id, id2vert = adjacency_matrix(g)
    used = np.zeros(n, dtype=np.bool_)
    vert_groups = []
    for i in range(n):
        j = vert2id[ord[i]]
        if ord[i] not in gadget_dict:
            vert_groups.append([j])
        elif not used[j]:
            u, v = gadget_dict[ord[i]]
            j1, j2 = vert2id[u], vert2id[v]
            vert_groups.append([j1, j2])
            used[j1] = used[j2] = True
    tree_edges = []
    sub_roots = []
    cur_root = n
    for vert_group in vert_groups:
        if len(vert_group) == 2:
            i, j = vert_group
            tree_edges.append((i, cur_root))
            tree_edges.append((cur_root, i))
            tree_edges.append((j, cur_root))
            tree_edges.append((cur_root, j))
            sub_roots.append(cur_root)
            cur_root += 1
        else:
            sub_roots.append(vert_group[0])
    assert len(sub_roots) >= 2
    if len(sub_roots) == 2:
        tree_edges.append((sub_roots[0], sub_roots[1]))
        tree_edges.append((sub_roots[1], sub_roots[0]))
        return tree_edges
    tree_edges.append((sub_roots[0], cur_root))
    tree_edges.append((cur_root, sub_roots[0]))
    tree_edges.append((sub_roots[-1], 2 * n - 3))
    tree_edges.append((2 * n - 3, sub_roots[-1]))
    for i in range(1, len(sub_roots) - 1):
        u = cur_root + i - 1
        v = sub_roots[i]
        tree_edges.append((u, v))
        tree_edges.append((v, u))
    for i in range(len(sub_roots) - 3):
        tree_edges.append((cur_root + i, cur_root + i + 1))
        tree_edges.append((cur_root + i + 1, cur_root + i))
    return tree_edges


def simulate_circuit(circ: zx.Circuit, state: str, effect: str) -> complex:
    g = circ.to_graph()
    zx.full_reduce(g)
    tree_edges = initial_rw_decomposition(g)
    assert len(tree_edges) == g.num_vertices() * 4 - 6
    _, vert2id, _ = adjacency_matrix(g)
    g.apply_state(state)
    g.apply_effect(effect)
    zx.clifford_simp(g)
    leaves = [vert2id[u] for u in g.vertices()]
    inc = incidence_list(tree_edges)
    tree_edges = induced_tree(inc, tree_edges, leaves)

    # TODO: simulated annealing
    # tree_edges = rw_decomposition(g)

    return simulate_graph(g, tree_edges)
