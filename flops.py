from pyzx import to_quimb_tensor
from rw_simulate import *
from quimb_helpers import *


def convolution_flops(r_u, r_v, r_w):
    r_max = max(r_u, r_v, r_w)
    if r_max == r_u:
        return 2 ** (r_v + r_w) * 3
    elif r_max == r_v:
        return 2 ** (r_u + r_w) * (4 + r_w)
    else:
        return 2 ** (r_u + r_v) * (4 + r_v)


def decomposition_flops(g, decomp):
    S_0 = set(g.vertices())

    def iterate(elem):
        if isinstance(elem, int):
            return {elem}, 0
        S_v, cost_v = iterate(elem[0])
        S_w, cost_w = iterate(elem[1])
        S_u = S_v | S_w
        r_u = rank_factorize(adjacency_matrix(g, S_u, S_0 - S_u))[0]
        r_v = rank_factorize(adjacency_matrix(g, S_v, S_0 - S_v))[0]
        r_w = rank_factorize(adjacency_matrix(g, S_w, S_0 - S_w))[0]
        return S_u, cost_v + cost_w + convolution_flops(r_u, r_v, r_w)

    if decomp is None:
        return 0
    return iterate(decomp)[1]


def rw_simulate_flops(circ: zx.Circuit, state: str, effect: str, **annealer_kwargs) -> int:
    g, decomp = initial_decomposition(circ, state, effect)
    g, decomp = improve_decomposition(g, decomp, **annealer_kwargs)
    return decomposition_flops(g, decomp)


def quimb_flops(circ: zx.Circuit, state: str, effect: str, optimize: str, initial: bool):
    if initial:
        qcirc = circuit2quimb(circ)
        reh = quimb_amplitude(qcirc, state, effect, optimize=optimize, rehearse=True)
        flops = reh['tree'].contraction_cost()
        return flops
    g = circuit2graph(circ, state, effect)
    g.apply_state('0' * circ.qubits)
    g.apply_effect('0' * circ.qubits)
    zx.full_reduce(g)
    net = to_quimb_tensor(g)
    flops = net.contraction_cost(optimize=optimize)
    return flops
