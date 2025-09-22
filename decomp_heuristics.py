import pyzx as zx
import quizx

from copy import deepcopy
from math import log2
from gf2 import REF
from graph import adjacency_matrix, build_linear, rank_width, rank_score_flops
from zx_helpers import phase_gadgets, simplify_get_pivots


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


def flow_order(g: zx.graph.base.BaseGraph, flatten=False):
    zx.to_graph_like(g)
    zx.id_simp(g)
    zx.spider_simp(g)
    layers, _ = causal_flow(g)
    init_order = sorted(list(layers.keys()), key=lambda v: layers[v], reverse=True)
    g.apply_state('0' * g.num_inputs())
    g.apply_effect('0' * g.num_outputs())
    g = g.copy(backend="quizx-vec")
    init_order_dict = {v: i for i, v in enumerate(sorted(init_order))}
    pivots = simplify_get_pivots(g)
    order = [init_order_dict[v] for v in init_order if init_order_dict[v] in g.vertices()]
    gadgets = phase_gadgets(g)
    for u, v in pivots + gadgets:
        if u not in order or v not in order:
            continue
        order.pop(order.index(v))
        order.insert(order.index(u) + 1, v)
    if not flatten:
        for u, v in gadgets:
            pos = order.index(u)
            if order[pos:pos + 2] != [u, v]:
                continue
            order[pos:pos + 2] = [[u, v]]
    return g, order


def linear_order(g: zx.graph.base.BaseGraph):
    n = g.num_vertices()
    if n == 0:
        return []
    vert = list(g.vertices())
    mat = adjacency_matrix(g, vert, vert)
    ref = REF(mat)
    ref.take(0)
    order = [vert[0]]
    for i in range(1, n):
        min_r, min_u = n, -1
        for u in ref.pivot_cols:
            cur_ref = deepcopy(ref)
            cur_ref.take(u)
            r = cur_ref.rank()
            if r < min_r:
                min_r = r
                min_u = u
        order.append(vert[min_u])
        ref.take(min_u)
    return order


def greedy_decomposition(g: zx.graph.base.BaseGraph):
    n = g.num_vertices()
    if n == 0:
        return None
    vert = list(g.vertices())
    mat = adjacency_matrix(g, vert, vert)
    refs = []
    decomps = []
    leaves = []
    for i in range(n):
        ref = REF(mat)
        ref.take(i)
        refs.append(ref)
        decomps.append(vert[i])
        leaves.append({i})
    refs_next = dict()
    edges = [set() for _ in range(n)]
    for i in range(n):
        for j in refs[i].pivot_cols:
            ref = deepcopy(refs[i])
            ref.take(j)
            refs_next[(i, j)] = refs_next[(j, i)] = ref
            edges[i].add(j)
            edges[j].add(i)
    i = 0
    for _ in range(n - 1):
        min_rank, i, j = min((ref.rank(), i, j) for (i, j), ref in refs_next.items())
        refs[i] = refs_next[(i, j)]
        refs[j] = None
        decomps[i] = [decomps[i], decomps[j]]
        decomps[j] = None
        leaves[i] |= leaves[j]
        leaves[j] = None
        edges_i = edges[i].copy()
        for k in edges_i:
            refs_next.pop((i, k))
            refs_next.pop((k, i))
            edges[i].remove(k)
            edges[k].remove(i)
        edges_j = edges[j].copy()
        for k in edges_j:
            refs_next.pop((j, k))
            refs_next.pop((k, j))
            edges[j].remove(k)
            edges[k].remove(j)
        for k in range(n):
            if refs[k] is None or k == i:
                continue
            if set(refs[i].pivot_cols) & leaves[k] or set(refs[k].pivot_cols) & leaves[i]:
                i1, i2 = (i, k) if len(leaves[i]) > len(leaves[k]) else (k, i)
                ref = deepcopy(refs[i1])
                for leaf in leaves[i2]:
                    ref.take(leaf)
                refs_next[(i1, i2)] = refs_next[(i2, i1)] = ref
                edges[i1].add(i2)
                edges[i2].add(i1)
    return decomps[i]


def dp_decomposition(g: zx.graph.base.BaseGraph, order):
    n = g.num_vertices()
    if n == 0:
        return None
    vert = list(g.vertices())
    order = [vert.index(v) for v in order]
    mat = adjacency_matrix(g, vert, vert)
    rank_seg = [[-1] * n for _ in range(n)]
    for i in range(n):
        ref = REF(mat)
        for j in range(i, n):
            ref.take(order[j])
            rank_seg[i][j] = ref.rank()
    dp = [[2 ** n] * n for _ in range(n)]
    ans = [[-1] * n for _ in range(n)]
    for i in range(n):
        dp[i][i] = 0
    for l in range(1, n):
        for i in range(n - l):
            j = i + l
            for k in range(i, j):
                r1, r2, r3 = rank_seg[i][k], rank_seg[k + 1][j], rank_seg[i][j]
                score = 2 ** (r1 + r2 + r3 - max(r1, r2, r3))
                if dp[i][j] > dp[i][k] + dp[k + 1][j] + score:
                    dp[i][j] = dp[i][k] + dp[k + 1][j] + score
                    ans[i][j] = k

    def restore_decomp(i, j):
        if i == j:
            return vert[order[i]]
        k = ans[i][j]
        return [restore_decomp(i, k), restore_decomp(k + 1, j)]

    return restore_decomp(0, n - 1)


def auto_decomposition(g: zx.graph.base.BaseGraph, order):
    n = g.num_vertices()
    vert = list(g.vertices())
    order = [vert.index(v) for v in order]
    mat = adjacency_matrix(g, vert, vert)
    refs = []
    cut_refs = []
    cut_ref = REF(mat)
    for v in order:
        ref = REF(mat)
        ref.take(v)
        refs.append(ref)
        cut_ref.take(v)
        cut_refs.append(deepcopy(cut_ref))

    decomps = [vert[v] for v in order]
    leaves = [{v} for v in order]
    scores = [0.0] * n
    loc = list(range(n))
    refs_next = dict()
    edges = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if order[j] not in refs[i].pivot_cols:
                continue
            ref = deepcopy(refs[i])
            ref.take(order[j])
            refs_next[(i, j)] = refs_next[(j, i)] = ref
            edges[i].add(j)
            edges[j].add(i)

    best_decomp = build_linear(decomps)
    best_score = rank_score_flops(best_decomp, g)
    while refs_next:
        min_rank, i, j = min((ref.rank(), i, j) for (i, j), ref in refs_next.items())
        if i > j:
            i, j = j, i

        new_cut_refs = []
        for k in range(loc[i], loc[j]):
            new_cut_ref = deepcopy(cut_refs[k])
            for leaf in leaves[loc[j]]:
                new_cut_ref.take(leaf)
            new_cut_refs.append(new_cut_ref)

        r_i, r_j = refs[loc[i]].rank(), refs[loc[j]].rank()
        merge_score = r_i + r_j + min_rank - max(r_i, r_j, min_rank)
        refs[loc[i]] = refs_next[(i, j)]
        refs.pop(loc[j])
        cut_refs[loc[i]:loc[j]] = new_cut_refs
        cut_refs.pop(loc[j])
        decomps[loc[i]] = [decomps[loc[i]], decomps[loc[j]]]
        decomps.pop(loc[j])
        leaves[loc[i]] |= leaves[loc[j]]
        leaves.pop(loc[j])
        scores[loc[i]] += scores[loc[j]] + 2.0 ** merge_score
        scores.pop(loc[j])
        loc[j] = None
        for k in range(j + 1, n):
            if loc[k] is not None:
                loc[k] -= 1
        linear_score = 0
        for k in range(1, len(refs)):
            r1, r2, r3 = cut_refs[k - 1].rank(), cut_refs[k].rank(), refs[k].rank()
            linear_score += 2.0 ** (r1 + r2 + r3 - max(r1, r2, r3))
        cur_score = log2(sum(scores) + linear_score)
        if cur_score < best_score:
            best_score = cur_score
            best_decomp = build_linear(decomps)
        edges_i = edges[i].copy()
        for k in edges_i:
            refs_next.pop((i, k))
            refs_next.pop((k, i))
            edges[i].remove(k)
            edges[k].remove(i)
        edges_j = edges[j].copy()
        for k in edges_j:
            refs_next.pop((j, k))
            refs_next.pop((k, j))
            edges[j].remove(k)
            edges[k].remove(j)
        for k in range(n):
            if loc[k] is None or k == i:
                continue
            if (set(refs[loc[i]].pivot_cols) & leaves[loc[k]] or
                    set(refs[loc[k]].pivot_cols) & leaves[loc[i]]):
                i1, i2 = (i, k) if len(leaves[loc[i]]) > len(leaves[loc[k]]) else (k, i)
                ref = deepcopy(refs[loc[i1]])
                for leaf in leaves[loc[i2]]:
                    ref.take(leaf)
                refs_next[(i1, i2)] = refs_next[(i2, i1)] = ref
                edges[i1].add(i2)
                edges[i2].add(i1)

    return best_decomp


def anneal_decomposition(g: zx.graph.base.BaseGraph, decomp, verbose=False, **annealer_kwargs):
    if decomp is None:
        return None
    init_decomp = quizx.DecompTree.from_list(decomp)
    if verbose:
        init_rw = init_decomp.rankwidth(g)
        init_score = init_decomp.rankwidth_score(g, 'flops')
        print(f'Initial rank-decomposition has width {init_rw} and score {init_score}')
    ann = quizx.RankwidthAnnealer(g, init_decomp=init_decomp, **annealer_kwargs)
    final_decomp = ann.run()
    if verbose:
        final_rw = final_decomp.rankwidth(g)
        final_score = final_decomp.rankwidth_score(g, 'flops')
        print(f'Final rank-decomposition has width {final_rw} and score {final_score}')
    return final_decomp.to_list()


def compute_decomposition(g: zx.graph.base.BaseGraph, opt='auto', verbose=False):
    if opt == 'flow':
        g, order = flow_order(g)
        decomp = build_linear(order)
    elif opt == 'greedy':
        g.apply_state('0' * g.num_inputs())
        g.apply_effect('0' * g.num_outputs())
        zx.full_reduce(g)
        decomp = greedy_decomposition(g)
    elif opt == 'linear':
        g.apply_state('0' * g.num_inputs())
        g.apply_effect('0' * g.num_outputs())
        zx.full_reduce(g)
        order = linear_order(g)
        decomp = build_linear(order)
    elif opt == 'dp':
        g.apply_state('0' * g.num_inputs())
        g.apply_effect('0' * g.num_outputs())
        zx.full_reduce(g)
        order = linear_order(g)
        decomp = dp_decomposition(g, order)
    elif opt == 'auto':
        g, order = flow_order(g, flatten=True)
        decomp = auto_decomposition(g, order)
    else:
        raise ValueError('Unknown optimiser')
    if verbose:
        width, score = rank_width(decomp, g), rank_score_flops(decomp, g)
        print(f'Rank-decomposition ({opt}) has width {width} and score {score:.3f}')
    return g, decomp
