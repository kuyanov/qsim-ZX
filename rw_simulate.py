import numpy as np
import pyzx as zx
from itertools import product
from math import *

from decomp_heuristics import compute_decomposition
from gf2 import rank_factorize, generalized_inverse
from graph import adjacency_matrix
from zx_helpers import circuit2graph


def mat_image(M: np.ndarray) -> np.ndarray:
    n, m = M.shape
    M_ints = M @ (1 << np.arange(m))
    vals = np.zeros(1, dtype=np.int64)
    for pos in range(n):
        vals = np.concatenate([vals, M_ints[pos] ^ vals])
    return vals


def apply_parity_map(Psi: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Apply parity map M to state Psi.
    :param Psi: numpy array of size 2^n
    :param M: binary matrix of shape (n, m)
    :return Phi: numpy array of size 2^m
    """
    m = M.shape[1]
    ys = mat_image(M)
    Phi = np.zeros(2 ** m, dtype=Psi.dtype)
    np.add.at(Phi, ys, Psi)
    Phi /= sqrt(2) ** (M.sum() - m)
    return Phi


def phase_tensor(E: np.ndarray):
    """
    Generate P_{a,b} = (-1)^{<a, Eb>} / sqrt(2)^|E|.
    :param E: binary matrix of shape (n, m)
    :return P: numpy array of size 2^{n+m}
    """
    n, m = E.shape
    Eb = mat_image(E.T)
    aEb = np.zeros((1, 2 ** m), dtype=np.int8)
    for pos in range(n):
        aEb = np.vstack([aEb, aEb ^ ((Eb >> pos) & 1)])
    P = (-1) ** aEb.reshape(-1, order='F') / sqrt(2) ** E.sum()
    return P


def conv_naive(Psi_v: np.ndarray, Psi_w: np.ndarray,
               E_vw: np.ndarray, E_vu: np.ndarray, E_wu: np.ndarray) -> np.ndarray:
    """
    Perform convolution naively in 2^{r_u + r_v + r_w} time.
    :param Psi_v: numpy array of size 2^{r_v}
    :param Psi_w: numpy array of size 2^{r_w}
    :param E_vw: binary matrix of shape (r_v, r_w)
    :param E_vu: binary matrix of shape (r_v, r_u)
    :param E_wu: binary matrix of shape (r_w, r_u)
    :return Psi_u: numpy array of size 2^{r_u}
    """
    r_u, r_v, r_w = E_vu.shape[1], E_vu.shape[0], E_wu.shape[0]
    Psi_v, Psi_w = Psi_v.reshape((2,) * r_v, order='F'), Psi_w.reshape((2,) * r_w, order='F')
    Psi_u_hat = np.zeros((2,) * r_u, dtype=Psi_v.dtype)
    for x in product(range(2), repeat=r_u):
        for a in product(range(2), repeat=r_v):
            for b in product(range(2), repeat=r_w):
                phase = np.dot(a, E_vu @ x) + np.dot(b, E_wu @ x) + np.dot(a, E_vw @ b)
                Psi_u_hat[x] += Psi_v[a] * Psi_w[b] * (-1) ** phase
    Psi_u = np.fft.fftn(Psi_u_hat)
    Psi_u /= sqrt(2) ** (E_vw.sum() + E_vu.sum() + E_wu.sum() + r_u)
    return Psi_u.reshape(-1, order='F')


def conv_vw(Psi_v: np.ndarray, Psi_w: np.ndarray,
            E_vw: np.ndarray, E_vu: np.ndarray, E_wu: np.ndarray) -> np.ndarray:
    """
    Perform convolution in 2^{r_v + r_w} time:
      1. Take Psi_v ⊗ Psi_w, multiply by phase tensor for E_vw
      2. Apply parity map [E_vu; E_wu]
    :param Psi_v: numpy array of size 2^{r_v}
    :param Psi_w: numpy array of size 2^{r_w}
    :param E_vw: binary matrix of shape (r_v, r_w)
    :param E_vu: binary matrix of shape (r_v, r_u)
    :param E_wu: binary matrix of shape (r_w, r_u)
    :return Psi_u: numpy array of size 2^{r_u}
    """
    Psi_vw = np.outer(Psi_v, Psi_w).reshape(-1, order='F')
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
    :param Psi_v: numpy array of size 2^{r_v}
    :param Psi_w: numpy array of size 2^{r_w}
    :param E_vw: binary matrix of shape (r_v, r_w)
    :param E_vu: binary matrix of shape (r_v, r_u)
    :param E_wu: binary matrix of shape (r_w, r_u)
    :return Psi_u: numpy array of size 2^{r_u}
    """
    r_u, r_v, r_w = E_vu.shape[1], E_vu.shape[0], E_wu.shape[0]
    E = np.hstack([E_vw.T, E_wu])
    Psi_vu = apply_parity_map(Psi_w, E).reshape((2,) * r_v + (2 ** r_u,), order='F')
    Psi_vu = (np.fft.fftn(Psi_vu, axes=tuple(range(r_v)))).reshape(-1, order='F') / sqrt(2) ** r_v
    E2 = np.block([[np.eye(r_v), E_vu],
                   [np.zeros((r_u, r_v)), np.eye(r_u)]]).astype(np.int8)
    Psi_vu = apply_parity_map(Psi_vu, E2)
    Psi_u = np.tensordot(Psi_v, Psi_vu.reshape((2 ** r_v, 2 ** r_u), order='F'), axes=(0, 0))
    return Psi_u


def conv_uw(Psi_v: np.ndarray, Psi_w: np.ndarray,
            E_vw: np.ndarray, E_vu: np.ndarray, E_wu: np.ndarray) -> np.ndarray:
    """
    Perform convolution in 2^{r_u + r_w} time by calling conv_uv.
    :param Psi_v: numpy array of size 2^{r_v}
    :param Psi_w: numpy array of size 2^{r_w}
    :param E_vw: binary matrix of shape (r_v, r_w)
    :param E_vu: binary matrix of shape (r_v, r_u)
    :param E_wu: binary matrix of shape (r_w, r_u)
    :return Psi_u: numpy array of size 2^{r_u}
    """
    return conv_uv(Psi_w, Psi_v, E_vw.T, E_wu, E_vu)


def conv(Psi_v: np.ndarray, Psi_w: np.ndarray,
         E_vw: np.ndarray, E_vu: np.ndarray, E_wu: np.ndarray, verbose=False) -> np.ndarray:
    """
    Convolution in time 2^{r_u + r_v + r_w - max(r_u, r_v, r_w)} by calling a suitable subroutine.
    :param Psi_v: numpy array of size 2^{r_v}
    :param Psi_w: numpy array of size 2^{r_w}
    :param E_vw: binary matrix of shape (r_v, r_w)
    :param E_vu: binary matrix of shape (r_v, r_u)
    :param E_wu: binary matrix of shape (r_w, r_u)
    :return Psi_u: numpy array of size 2^{r_u}
    """
    r_u, r_v, r_w = E_vu.shape[1], E_vu.shape[0], E_wu.shape[0]
    if verbose:
        print('conv', r_u, r_v, r_w)
    r_max = max(r_u, r_v, r_w)
    if r_u == r_max:
        return conv_vw(Psi_v, Psi_w, E_vw, E_vu, E_wu)
    elif r_v == r_max:
        return conv_uw(Psi_v, Psi_w, E_vw, E_vu, E_wu)
    else:
        return conv_uv(Psi_v, Psi_w, E_vw, E_vu, E_wu)


def simulate_graph(g: zx.graph.base.BaseGraph, decomp, preserve_scalar=True, verbose=False) -> complex:
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
        Cout = np.concatenate((M_v, M_w))
        Cout[:, S_v | S_w] = 0

        r_u, U, V = rank_factorize(Cout)
        r_v, r_w = M_v.shape[0], M_w.shape[0]
        U, V = U[:, :r_u], V[:r_u]
        E_vu, E_wu = U[:r_v], U[r_v:]
        M_u = V

        B = adjacency_matrix(g, v_all[S_v], v_all[S_w])
        Bg = generalized_inverse(B)
        E_vw = (Cin_v @ Bg @ Cin_w.T) % 2
        Psi_u = conv(Psi_v, Psi_w, E_vw, E_vu, E_wu, verbose=verbose)
        pw = Cout.sum() - U.sum() - V.sum() + r_u
        pw += Cin_v.sum() + Cin_w.sum() - E_vw.sum() - B.sum()
        Psi_u /= sqrt(2) ** pw
        return S_v | S_w, Psi_u, M_u

    if decomp is None:
        return g.scalar.to_number() ** preserve_scalar
    return simulate_recursive(decomp)[1].item() * g.scalar.to_number() ** preserve_scalar


def simulate_circuit(circ: zx.Circuit, state: str, effect: str, opt='auto', verbose=False) -> complex:
    if verbose:
        print(f'Simulating quantum circuit with {circ.qubits} qubits and {len(circ.gates)} gates')
    g = circuit2graph(circ, state, effect)
    g, decomp = compute_decomposition(g, opt=opt, verbose=verbose)
    # decomp = anneal_decomposition(g, decomp, verbose=verbose, **annealer_kwargs)
    return simulate_graph(g, decomp, verbose=verbose)
