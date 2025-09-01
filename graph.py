import quizx
from galois import GF2
from typing import Iterable

from gf2 import rank_factorize


def adjacency_matrix(g: quizx.VecGraph, v_left: Iterable[int], v_right: Iterable[int]) -> GF2:
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


def rank_width(decomp, g: quizx.VecGraph) -> int:
    v_all = set(g.vertices())
    ranks = []

    def iterate(elem):
        if isinstance(elem, int):
            v_left = {elem}
        else:
            s1, s2 = iterate(elem[0]), iterate(elem[1])
            v_left = s1 | s2
            assert len(v_left) == len(s1) + len(s2)
        v_right = v_all - v_left
        mat = adjacency_matrix(g, v_left, v_right)
        ranks.append(rank_factorize(mat)[0])
        return v_left

    assert iterate(decomp) == v_all
    return max(ranks)
