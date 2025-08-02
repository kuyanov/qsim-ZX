import pyzx as zx
from galois import GF2
from typing import Iterable
from quizx.quizx import Scalar


def adjacency_matrix(g: zx.graph.base.BaseGraph, v_left: Iterable[int], v_right: Iterable[int]) -> GF2:
    left_id = {u: i for i, u in enumerate(v_left)}
    right_id = {v: i for i, v in enumerate(v_right)}
    n, m = len(left_id), len(right_id)
    mat = GF2.Zeros((n, m))
    for u, v in g.edge_set():
        if u in left_id and v in right_id:
            mat[left_id[u]][right_id[v]] = 1
        if v in left_id and u in right_id:
            mat[left_id[v]][right_id[u]] = 1
    return mat


def to_quizx(g_orig):
    g = zx.Graph('quizx-vec')
    g.scalar = Scalar.complex(g_orig.scalar.to_number())
    ty = g_orig.types()
    ph = g_orig.phases()
    qs = g_orig.qubits()
    rs = g_orig.rows()
    vtab = dict()
    for v in g_orig.vertices():
        i = g.add_vertex(ty[v], phase=ph[v])
        if v in qs: g.set_qubit(i, qs[v])
        if v in rs:
            g.set_row(i, rs[v])
        vtab[v] = i
        for k in g_orig.vdata_keys(v):
            g.set_vdata(i, k, g_orig.vdata(v, k))
    for v in g_orig.grounds():
        g.set_ground(vtab[v], True)

    new_inputs = tuple(vtab[i] for i in g_orig.inputs())
    new_outputs = tuple(vtab[i] for i in g_orig.outputs())
    g.set_inputs(new_inputs)
    g.set_outputs(new_outputs)

    for e in g_orig.edges():
        s, t = g_orig.edge_st(e)
        g.add_edge((vtab[s], vtab[t]), g_orig.edge_type(e))

    return g
