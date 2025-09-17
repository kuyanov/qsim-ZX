import pyzx as zx

from zx_helpers import circuit2graph


def quimb_amplitude(circ: zx.Circuit, state: str, effect: str, optimize='auto', initial=False):
    g = circuit2graph(circ, state, effect)
    g.apply_state('0' * circ.qubits)
    g.apply_effect('0' * circ.qubits)
    if not initial:
        zx.full_reduce(g)
    net = zx.to_quimb_tensor(g).full_simplify()
    return net.contract(output_inds=(), optimize=optimize)
