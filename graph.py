import math
import numpy as np
import pyzx as zx
from copy import deepcopy
from typing import Iterable

from gf2 import REF


def adjacency_matrix(g: zx.graph.base.BaseGraph, v_left: Iterable[int], v_right: Iterable[int]) -> np.ndarray:
    left_id = {u: i for i, u in enumerate(v_left)}
    right_id = {v: i for i, v in enumerate(v_right)}
    n, m = len(left_id), len(right_id)
    mat = np.zeros((n, m), dtype=np.int8)
    for u, v in g.edge_set():
        if u in left_id and v in right_id:
            mat[left_id[u]][right_id[v]] = 1
        if v in left_id and u in right_id:
            mat[left_id[v]][right_id[u]] = 1
    return mat


def build_linear(arr: Iterable):
    decomp = None
    for elem in arr:
        decomp = elem if decomp is None else [decomp, elem]
    return decomp


def calc_ranks(decomp, g: zx.graph.base.BaseGraph):
    v_all = list(g.vertices())
    v_id = {v: i for i, v in enumerate(v_all)}
    mat = adjacency_matrix(g, v_all, v_all)

    def iterate(elem):
        if isinstance(elem, int):
            ref = REF(mat)
            ref.take(v_id[elem])
            leaves = {v_id[elem]}
            return elem, ref, leaves
        child1, ref1, leaves1 = iterate(elem[0])
        child2, ref2, leaves2 = iterate(elem[1])
        if len(leaves1) >= len(leaves2):
            ref = deepcopy(ref1)
            for j in leaves2:
                ref.take(j)
        else:
            ref = deepcopy(ref2)
            for j in leaves1:
                ref.take(j)
        return [(child1, ref1, leaves1), (child2, ref2, leaves2)], ref, leaves1 | leaves2

    result = iterate(decomp)
    assert result[2] == set(range(g.num_vertices()))
    return result


def rank_width(decomp, g: zx.graph.base.BaseGraph, calc_rs=True) -> int:
    def iterate(data):
        if isinstance(data[0], int):
            return data[1].rank()
        return max(data[1].rank(), iterate(data[0][0]), iterate(data[0][1]))

    if decomp is None:
        return 0
    return iterate(calc_ranks(decomp, g) if calc_rs else decomp)


def rank_score_flops(decomp, g: zx.graph.base.BaseGraph, calc_rs=True) -> float:
    def iterate(data):
        if isinstance(data[0], int):
            return 0
        score1 = iterate(data[0][0])
        score2 = iterate(data[0][1])
        r1, r2, r3 = data[0][0][1].rank(), data[0][1][1].rank(), data[1].rank()
        return score1 + score2 + 2 ** (r1 + r2 + r3 - max(r1, r2, r3))

    if decomp is None:
        return 0
    score = iterate(calc_ranks(decomp, g) if calc_rs else decomp)
    return math.log2(max(score, 1))


def rank_score_square(decomp, g: zx.graph.base.BaseGraph, calc_rs=True) -> int:
    def iterate(data):
        if isinstance(data[0], int):
            return data[1].rank() ** 2
        score1 = iterate(data[0][0])
        score2 = iterate(data[0][1])
        return score1 + score2 + data[1].rank() ** 2

    if decomp is None:
        return 0
    score = iterate(calc_ranks(decomp, g) if calc_rs else decomp)
    return score ** 0.5
