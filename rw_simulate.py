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


def apply_parity_map(Psi: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Apply parity map M to state Psi.
    :param Psi: numpy array of shape (2,) * n
    :param M: binary matrix of shape (n, m)
    :return Phi: numpy array of shape (2,) * m
    """
    n, m = M.shape
    M_ints = M.T @ (1 << np.arange(n))
    Psi_flat = Psi.reshape(-1, order='F')
    xs = np.arange(2 ** n)
    ys = (np.bitwise_count(xs[:, None] & M_ints[None, :]) % 2) @ (1 << np.arange(m))
    Phi = np.zeros(2 ** m, dtype=Psi.dtype)
    np.add.at(Phi, ys, Psi_flat)
    Phi /= sqrt(2) ** (M.sum() - m)
    Phi = Phi.reshape((2,) * m, order='F')
    return Phi


def phase_tensor(E: np.ndarray):
    """
    Generate P_{a,b} = (-1)^{<a, Eb>} / sqrt(2)^|E|.
    :param E: binary matrix of shape (n, m)
    :return P: numpy array of shape (2,) * (n + m)
    """
    n, m = E.shape
    E_ints = E @ (1 << np.arange(m))
    a, b = np.arange(2 ** n), np.arange(2 ** m)
    Eb = (np.bitwise_count(b[:, None] & E_ints[None, :]) % 2) @ (1 << np.arange(n))
    inner = (np.bitwise_count(a[:, None] & Eb[None, :]) % 2).astype(np.int8)
    P = (-1) ** inner / sqrt(2) ** E.sum()
    P = P.reshape((2,) * (n + m), order='F')
    return P


def conv_naive(Psi_v: np.ndarray, Psi_w: np.ndarray,
               E_vw: np.ndarray, E_vu: np.ndarray, E_wu: np.ndarray) -> np.ndarray:
    """
    Perform convolution naively in 2^{r_u + r_v + r_w} time.
    :param Psi_v: numpy array of shape (2,) * r_v
    :param Psi_w: numpy array of shape (2,) * r_w
    :param E_vw: binary matrix of shape (r_v, r_w)
    :param E_vu: binary matrix of shape (r_v, r_u)
    :param E_wu: binary matrix of shape (r_w, r_u)
    :return Psi_u: numpy array of shape (2,) * r_u
    """
    r_u, r_v, r_w = E_vu.shape[1], Psi_v.ndim, Psi_w.ndim
    E2_vw, E2_vu, E2_wu = GF2(E_vw), GF2(E_vu), GF2(E_wu)
    Psi_u_hat = np.zeros((2,) * r_u, dtype=Psi_v.dtype)
    for x in product(range(2), repeat=r_u):
        for a in product(range(2), repeat=r_v):
            for b in product(range(2), repeat=r_w):
                x1, a1, b1 = GF2(x), GF2(a), GF2(b)
                phase = int(np.dot(a1, E2_vu @ x1) + np.dot(b1, E2_wu @ x1) + np.dot(a1, E2_vw @ b1))
                Psi_u_hat[x] += Psi_v[a] * Psi_w[b] * (-1) ** phase
    Psi_u = np.fft.fftn(Psi_u_hat)
    Psi_u /= sqrt(2) ** (E_vw.sum() + E_vu.sum() + E_wu.sum() + r_u)
    return Psi_u


def conv_vw(Psi_v: np.ndarray, Psi_w: np.ndarray,
            E_vw: np.ndarray, E_vu: np.ndarray, E_wu: np.ndarray) -> np.ndarray:
    """
    Perform convolution in 2^{r_v + r_w} time:
      1. Take Psi_v ⊗ Psi_w, multiply by phase tensor for E_vw
      2. Apply parity map [E_vu; E_wu]
    :param Psi_v: numpy array of shape (2,) * r_v
    :param Psi_w: numpy array of shape (2,) * r_w
    :param E_vw: binary matrix of shape (r_v, r_w)
    :param E_vu: binary matrix of shape (r_v, r_u)
    :param E_wu: binary matrix of shape (r_w, r_u)
    :return Psi_u: numpy array of shape (2,) * r_u
    """
    Psi_vw = np.tensordot(Psi_v, Psi_w, axes=0)
    Psi_vw *= phase_tensor(E_vw)
    E = np.vstack([E_vu, E_wu])
    Psi_u = apply_parity_map(Psi_vw, E)
    return Psi_u


def conv_uv(Psi_v: np.ndarray, Psi_w: np.ndarray,
            E_vw: np.ndarray, E_vu: np.ndarray, E_wu: np.ndarray) -> np.ndarray:
    """
    Perform convolution in 2^{r_u + r_v} time:
      1. Apply parity map [E_vw^T, E_wu] to Psi_w
      2. Multiply by phase tensor for E_vu
      3. Post-select with Psi_v and apply FT on second part
    :param Psi_v: numpy array of shape (2,) * r_v
    :param Psi_w: numpy array of shape (2,) * r_w
    :param E_vw: binary matrix of shape (r_v, r_w)
    :param E_vu: binary matrix of shape (r_v, r_u)
    :param E_wu: binary matrix of shape (r_w, r_u)
    :return Psi_u: numpy array of shape (2,) * r_u
    """
    r_u, r_v, r_w = E_vu.shape[1], Psi_v.ndim, Psi_w.ndim
    E = np.hstack([E_vw.T, E_wu])
    Psi_vu = apply_parity_map(Psi_w, E)
    Psi_vu = np.fft.fftn(Psi_vu, axes=tuple(range(r_v))) / sqrt(2) ** r_v
    E2 = np.block([[np.eye(r_v), E_vu],
                   [np.zeros((r_u, r_v)), np.eye(r_u)]]).astype(bool)
    Psi_vu = apply_parity_map(Psi_vu, E2)
    Psi_u = np.tensordot(Psi_v, Psi_vu, axes=(tuple(range(r_v)), tuple(range(r_v))))
    return Psi_u


def conv_uw(Psi_v: np.ndarray, Psi_w: np.ndarray,
            E_vw: np.ndarray, E_vu: np.ndarray, E_wu: np.ndarray) -> np.ndarray:
    """
    Perform convolution in 2^{r_u + r_w} time by calling conv_uv.
    :param Psi_v: numpy array of shape (2,) * r_v
    :param Psi_w: numpy array of shape (2,) * r_w
    :param E_vw: binary matrix of shape (r_v, r_w)
    :param E_vu: binary matrix of shape (r_v, r_u)
    :param E_wu: binary matrix of shape (r_w, r_u)
    :return Psi_u: numpy array of shape (2,) * r_u
    """
    return conv_uv(Psi_w, Psi_v, E_vw.T, E_wu, E_vu)


def conv(Psi_v: np.ndarray, Psi_w: np.ndarray,
         E_vw: np.ndarray, E_vu: np.ndarray, E_wu: np.ndarray) -> np.ndarray:
    """
    Convolution in time 2^{r_u + r_v + r_w - max(r_u, r_v, r_w)} by calling a suitable subroutine.
    :param Psi_v: numpy array of shape (2,) * r_v
    :param Psi_w: numpy array of shape (2,) * r_w
    :param E_vw: binary matrix of shape (r_v, r_w)
    :param E_vu: binary matrix of shape (r_v, r_u)
    :param E_wu: binary matrix of shape (r_w, r_u)
    :return Psi_u: numpy array of shape (2,) * r_u
    """
    r_u, r_v, r_w = E_vu.shape[1], Psi_v.ndim, Psi_w.ndim
    r_max = max(r_u, r_v, r_w)
    if r_u == r_max:
        return conv_vw(Psi_v, Psi_w, E_vw, E_vu, E_wu)
    elif r_v == r_max:
        return conv_uw(Psi_v, Psi_w, E_vw, E_vu, E_wu)
    else:
        return conv_uv(Psi_v, Psi_w, E_vw, E_vu, E_wu)


def simulate_graph(g: zx.graph.base.BaseGraph, decomp, preserve_scalar=True) -> complex:
    n = g.num_vertices()
    v_all = np.array(list(g.vertices()))
    v_id = {v: i for i, v in enumerate(v_all)}

    def simulate_recursive(elem):
        if isinstance(elem, int):
            phase = g.phase(elem)
            Psi_u = np.array([1, np.exp(pi * 1j * phase)])
            M_u = adjacency_matrix(g, [elem], v_all)
            S_u = np.zeros(n, dtype=bool)
            S_u[v_id[elem]] = True
            return S_u, Psi_u, M_u

        S_v, Psi_v, M_v = simulate_recursive(elem[0])
        S_w, Psi_w, M_w = simulate_recursive(elem[1])
        Cin_v = M_v[:, S_w]
        Cin_w = M_w[:, S_v]
        Cout = GF2(np.concatenate((M_v, M_w)))
        Cout[:, S_v | S_w] = 0

        r_u, U, V = rank_factorize(Cout)
        r_v, r_w = Psi_v.ndim, Psi_w.ndim
        U, V = U[:, :r_u], V[:r_u]
        E_vu, E_wu = U[:r_v] == 1, U[r_v:] == 1
        M_u = V

        B = adjacency_matrix(g, v_all[S_v], v_all[S_w])
        Bg = generalized_inverse(B)
        E_vw = (Cin_v @ Bg @ Cin_w.T) == 1
        Psi_u = conv(Psi_v, Psi_w, E_vw, E_vu, E_wu)
        pw = weight(Cout) - weight(U) - weight(V) + r_u
        pw += weight(Cin_v) + weight(Cin_w) - weight(E_vw) - weight(B)
        Psi_u /= sqrt(2) ** pw
        return S_v | S_w, Psi_u, M_u

    if n == 0:
        return g.scalar.to_number() ** preserve_scalar
    return simulate_recursive(decomp)[1].item() * g.scalar.to_number() ** preserve_scalar


def simulate_circuit(circ: zx.Circuit, state: str, effect: str, verbose=False) -> complex:
    g = circ.to_graph()
    zx.full_reduce(g)
    decomp = pauli_flow_decomposition(g)
    g.apply_state(state)
    g.apply_effect(effect)
    zx.clifford_simp(g)
    if verbose:
        print(f'Graph-like diagram has {g.num_vertices()} vertices and {g.num_edges()} edges')
    if g.num_vertices() == 0:
        return simulate_graph(g, [])
    g2 = g.copy(backend="quizx-vec")
    decomp = sub_decomposition(decomp, list(g.vertices()), list(g2.vertices()))
    init_decomp = quizx.DecompTree.from_list(decomp)
    if verbose:
        init_rw = init_decomp.rankwidth(g2)
        init_score = init_decomp.rankwidth_score(g2, 'flops')
        print(f'Initial rank-decomposition has width {init_rw} and score_flops = {init_score}')
    ann = quizx.RankwidthAnnealer(g2, init_decomp=init_decomp)
    final_decomp = ann.run()
    if verbose:
        final_rw = final_decomp.rankwidth(g2)
        final_score = final_decomp.rankwidth_score(g2, 'flops')
        print(f'Final rank-decomposition has width {final_rw} and score_flops = {final_score}')
    return simulate_graph(g2, final_decomp.to_list())
