import sys
import os

import numpy as np
import pyzx as zx
import quizx
from galois import GF2
from itertools import product
from math import *
from pyzx.gflow import gflow

from gf2 import rank_factorize, generalized_inverse
from graph import adjacency_matrix, rank_width


def weight(A: GF2) -> int:
    return int((A == 1).sum())


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
                   [np.zeros((r_u, r_v)), np.eye(r_u)]]).astype(bool)
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


def simulate_graph(g: quizx.VecGraph, decomp, preserve_scalar=True, verbose=False) -> complex:
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
        r_v, r_w = M_v.shape[0], M_w.shape[0]
        U, V = U[:, :r_u], V[:r_u]
        E_vu, E_wu = U[:r_v] == 1, U[r_v:] == 1
        M_u = V

        B = adjacency_matrix(g, v_all[S_v], v_all[S_w])
        Bg = generalized_inverse(B)
        E_vw = (Cin_v @ Bg @ Cin_w.T) == 1
        Psi_u = conv(Psi_v, Psi_w, E_vw, E_vu, E_wu, verbose=verbose)
        pw = weight(Cout) - weight(U) - weight(V) + r_u
        pw += weight(Cin_v) + weight(Cin_w) - weight(E_vw) - weight(B)
        Psi_u /= sqrt(2) ** pw
        return S_v | S_w, Psi_u, M_u

    if decomp is None:
        return g.scalar.to_number() ** preserve_scalar
    return simulate_recursive(decomp)[1].item() * g.scalar.to_number() ** preserve_scalar


def pauli_flow(g: zx.graph.base.BaseGraph):
    layers = gflow(g, pauli=True)[0]
    order = (list(g.inputs()) +
             sorted(list(layers.keys()), key=lambda v: layers[v]) +
             list(g.outputs()))
    return order


def causal_flow(g: zx.graph.base.BaseGraph):
    l = dict()
    nxt = dict()
    V = set(g.vertices())
    In = set(g.inputs())
    Inc = V - In
    Out = set(g.outputs())
    Outc = V - Out
    C = set(g.outputs())
    k = 1
    while True:
        Out1 = set()
        C1 = set()
        for v in C:
            nb = set(g.neighbors(v)) & Outc
            if len(nb) == 1:
                u = list(nb)[0]
                nxt[u] = v
                l[v] = k
                Out1 |= {u}
                C1 |= {v}
        if not Out1:
            if Out == V:
                for v in In:
                    l[v] = k
                return l, nxt
            return None
        Out |= Out1
        Outc -= Out1
        C = (C - C1) | (Out1 & Inc)
        k += 1


def phase_gadgets(g: zx.graph.base.BaseGraph):
    gadgets = []
    for v in g.vertices():
        if g.vertex_degree(v) != 1:
            continue
        u = list(g.neighbors(v))[0]
        if v in g.inputs() or v in g.outputs() or g.phase(u) != 0:
            continue
        gadgets.append((v, u))
    return gadgets


class capture_fd_stdout:
    """
    Context manager that captures ALL writes to the process's stdout (fd=1),
    including from Rust/C extensions using printf/println!.
    """

    def __enter__(self):
        self._saved_fd = os.dup(1)  # duplicate stdout fd
        self._r, self._w = os.pipe()  # create pipe
        os.dup2(self._w, 1)  # redirect fd=1 to pipe writer
        os.close(self._w)  # close our duplicate writer
        return self

    def __exit__(self, exc_type, exc, tb):
        os.dup2(self._saved_fd, 1)  # restore stdout
        os.close(self._saved_fd)  # close saved copy

    def read(self) -> str:
        # Read all data from the pipe reader
        chunks = []
        # Set non-blocking read in case nothing is there
        import fcntl
        flags = fcntl.fcntl(self._r, fcntl.F_GETFL)
        fcntl.fcntl(self._r, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        try:
            while True:
                try:
                    data = os.read(self._r, 65536)
                    if not data:
                        break
                    chunks.append(data)
                except BlockingIOError:
                    break
        finally:
            os.close(self._r)
        return b"".join(chunks).decode(errors="replace")


def circuit2graph(circ: zx.Circuit, state: str = None, effect: str = None):
    new_circ = zx.Circuit(circ.qubits)
    gate_map = {
        '0': ['H', 'H'],
        '1': ['NOT'],
        '+': ['H'],
        '-': ['NOT', 'H'],
        'T': ['H', 'T']
    }
    if state:
        for i, ch in enumerate(state):
            new_circ.add_gates(' '.join(gate_map[ch]), i)
    new_circ.add_circuit(circ)
    if effect:
        for i, ch in enumerate(effect):
            new_circ.add_gates(' '.join(gate_map[ch][::-1]), i)
    g = new_circ.to_graph()
    return g


def initial_decomposition_old(circ: zx.Circuit, state: str, effect: str, verbose=False):
    g = circuit2graph(circ, state, effect)
    zx.full_reduce(g)
    if verbose:
        print(f'Reduced diagram has {g.num_vertices()} vertices and {g.num_edges()} edges')
    pauli_order = pauli_flow(g)
    g.apply_state('0' * circ.qubits)
    g.apply_effect('0' * circ.qubits)
    g2 = g.copy(backend="quizx-vec")
    order_dict = {v: i for i, v in enumerate(sorted(pauli_order))}
    gadgets_g = [(order_dict[elem[0]], order_dict[elem[1]]) for elem in phase_gadgets(g)]

    sys.stdout.flush()
    with capture_fd_stdout() as cap:
        quizx.full_simp(g2)
    pivots = [tuple(map(int, l.split())) for l in cap.read().splitlines() if l.strip()]

    if verbose:
        print(f'Final graph has {g2.num_vertices()} vertices and {g2.num_edges()} edges')

    if g2.num_vertices() == 0:
        return g2, None

    new_order = [order_dict[v] for v in pauli_order if order_dict[v] in g2.vertices()]
    gadgets_g2 = phase_gadgets(g2)
    for u, v in gadgets_g + pivots + gadgets_g2:
        if u not in new_order or v not in new_order:
            continue
        new_order.pop(new_order.index(v))
        new_order.insert(new_order.index(u) + 1, v)
    for u, v in gadgets_g2:
        pos = new_order.index(u)
        if new_order[pos + 1] != v:
            continue
        new_order[pos:pos + 2] = [[u, v]]

    decomp = new_order[0]
    for elem in new_order[1:]:
        decomp = [decomp, elem]
    return g2, decomp


def initial_decomposition(circ: zx.Circuit, state: str, effect: str, verbose=False):
    g = circuit2graph(circ, state, effect)
    zx.to_graph_like(g)
    zx.id_simp(g)
    zx.spider_simp(g)
    if verbose:
        print(f'Initial graph-like diagram has {g.num_vertices()} vertices and {g.num_edges()} edges')
    layers, _ = causal_flow(g)
    order = sorted(list(layers.keys()), key=lambda v: layers[v], reverse=True)

    g.apply_state('0' * circ.qubits)
    g.apply_effect('0' * circ.qubits)
    g2 = g.copy(backend="quizx-vec")
    order_dict = {v: i for i, v in enumerate(sorted(order))}

    sys.stdout.flush()
    with capture_fd_stdout() as cap:
        quizx.full_simp(g2)
    pivots = [tuple(map(int, l.split())) for l in cap.read().splitlines() if l.strip()]

    if verbose:
        print(f'Final graph-like diagram has {g2.num_vertices()} vertices and {g2.num_edges()} edges')

    if g2.num_vertices() == 0:
        return g2, None

    new_order = [order_dict[v] for v in order if order_dict[v] in g2.vertices()]
    gadgets_g2 = phase_gadgets(g2)
    for u, v in pivots + gadgets_g2:
        if u not in new_order or v not in new_order:
            continue
        new_order.pop(new_order.index(v))
        new_order.insert(new_order.index(u) + 1, v)
    for u, v in gadgets_g2:
        pos = new_order.index(u)
        if new_order[pos:pos + 2] != [u, v]:
            continue
        new_order[pos:pos + 2] = [[u, v]]
    decomp = new_order[0]
    for elem in new_order[1:]:
        decomp = [decomp, elem]
    return g2, decomp


def improve_decomposition(g: zx.graph.base.BaseGraph, decomp, verbose=False, **annealer_kwargs):
    if decomp is None:
        return g, None
    init_decomp = quizx.DecompTree.from_list(decomp)
    if verbose:
        init_rw = init_decomp.rankwidth(g)
        init_score = init_decomp.rankwidth_score(g, 'flops')
        print(f'Initial rank-decomposition has width {init_rw} and score_flops = {init_score}')
    ann = quizx.RankwidthAnnealer(g, init_decomp=init_decomp, **annealer_kwargs)
    final_decomp = ann.run()
    if verbose:
        final_rw = final_decomp.rankwidth(g)
        final_score = final_decomp.rankwidth_score(g, 'flops')
        print(f'Final rank-decomposition has width {final_rw} and score_flops = {final_score}')
    return g, final_decomp.to_list()


def simulate_circuit(circ: zx.Circuit, state: str, effect: str, verbose=False, **annealer_kwargs) -> complex:
    if verbose:
        print(f'Simulating quantum circuit with {circ.qubits} qubits and {len(circ.gates)} gates')
    g, decomp = initial_decomposition(circ, state, effect, verbose=verbose)
    g, decomp = improve_decomposition(g, decomp, verbose=verbose, **annealer_kwargs)
    return simulate_graph(g, decomp, verbose=verbose)
