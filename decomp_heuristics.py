import pyzx as zx
import quizx

from copy import deepcopy
from gf2 import REF
from graph import adjacency_matrix, rank_width, rank_score_flops
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


def flow_decomposition(g: zx.graph.base.BaseGraph, verbose=False):
    zx.to_graph_like(g)
    zx.id_simp(g)
    zx.spider_simp(g)
    if verbose:
        print(f'Initial graph-like diagram has {g.num_vertices()} vertices and {g.num_edges()} edges')
    layers, _ = causal_flow(g)
    order = sorted(list(layers.keys()), key=lambda v: layers[v], reverse=True)

    g.apply_state('0' * g.num_inputs())
    g.apply_effect('0' * g.num_outputs())
    g2 = g.copy(backend="quizx-vec")
    order_dict = {v: i for i, v in enumerate(sorted(order))}
    pivots = simplify_get_pivots(g2)

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


def linear_order(g: zx.graph.base.BaseGraph):
    n = g.num_vertices()
    vert = list(g.vertices())
    mat = adjacency_matrix(g, vert, vert)
    ref = REF(mat)
    ref.take(0)
    order = [0]
    for i in range(1, n):
        min_r, min_u = n, -1
        for u in ref.pivot_cols:
            cur_ref = deepcopy(ref)
            cur_ref.take(u)
            r = cur_ref.rank()
            if r < min_r:
                min_r = r
                min_u = u
        order.append(min_u)
        ref.take(min_u)
    return order


def linear_decomposition(g: zx.graph.base.BaseGraph, verbose=False):
    vert = list(g.vertices())
    order = linear_order(g)
    decomp = vert[order[0]]
    for i in range(1, len(order)):
        decomp = [decomp, vert[order[i]]]
    if verbose:
        width = rank_width(decomp, g)
        score = rank_score_flops(decomp, g)
        print(f'Linear rank-decomposition has width {width} and score {score:.3f}')
    return decomp


def greedy_decomposition(g: zx.graph.base.BaseGraph, verbose=False):
    n = g.num_vertices()
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
    decomp = decomps[i]
    if verbose:
        width = rank_width(decomp, g)
        score = rank_score_flops(decomp, g)
        print(f'Greedy rank-decomposition has width {width} and score {score:.3f}')
    return decomp


def dp_decomposition(g: zx.graph.base.BaseGraph, verbose=False):
    n = g.num_vertices()
    vert = list(g.vertices())
    mat = adjacency_matrix(g, vert, vert)
    order = linear_order(g)

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

    decomp = restore_decomp(0, n - 1)
    if verbose:
        width = rank_width(decomp, g)
        score = rank_score_flops(decomp, g)
        print(f'DP rank-decomposition has width {width} and score {score:.3f}')
    return decomp


def auto_decomposition(g: zx.graph.base.BaseGraph, verbose=False):
    n = g.num_vertices()
    vert = list(g.vertices())
    mat = adjacency_matrix(g, vert, vert)
    order = linear_order(g)
    cut_refs = []
    ref = REF(mat)
    max_width = 0
    for i in order:
        ref.take(i)
        cut_refs.append(deepcopy(ref))
        max_width = max(max_width, ref.rank())

    refs = []
    decomps = []
    leaves = []
    for i in order:
        ref = REF(mat)
        ref.take(i)
        refs.append(ref)
        decomps.append(vert[i])
        leaves.append({i})
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

    while refs_next:
        min_rank, i, j = min((ref.rank(), i, j) for (i, j), ref in refs_next.items())
        if i > j:
            i, j = j, i
        r1, r2, r3 = refs[i].rank(), refs[j].rank(), min_rank
        if r1 + r2 + r3 - max(r1, r2, r3) > max_width + 1:
            refs_next.pop((i, j))
            refs_next.pop((j, i))
            edges[i].remove(j)
            edges[j].remove(i)
            continue

        ranks = [refs[k].rank() if k != i else min_rank for k in range(n) if refs[k] and k != j]
        new_cut_refs = deepcopy(cut_refs)
        for k in range(i, j):
            if new_cut_refs[k]:
                for leaf in leaves[j]:
                    new_cut_refs[k].take(leaf)
        new_cut_refs[j] = None
        cut_ranks = [new_cut_refs[k].rank() for k in range(n) if new_cut_refs[k]]
        assert len(ranks) == len(cut_ranks)
        bad = False
        for k in range(1, len(cut_ranks)):
            r1, r2, r3 = cut_ranks[k - 1], cut_ranks[k], ranks[k]
            if r1 + r2 + r3 - max(r1, r2, r3) > max_width + 1:
                bad = True
                break
        if bad:
            refs_next.pop((i, j))
            refs_next.pop((j, i))
            edges[i].remove(j)
            edges[j].remove(i)
            continue

        refs[i] = refs_next[(i, j)]
        refs[j] = None
        cut_refs = new_cut_refs
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

    decomp = None
    for i in range(n):
        if decomps[i] is not None:
            decomp = [decomp, decomps[i]] if decomp is not None else decomps[i]
    if verbose:
        width = rank_width(decomp, g)
        score = rank_score_flops(decomp, g)
        print(f'Auto rank-decomposition has width {width} and score {score:.3f}')
    return decomp


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
        return flow_decomposition(g, verbose=verbose)
    g.apply_state('0' * g.num_inputs())
    g.apply_effect('0' * g.num_outputs())
    zx.full_reduce(g)
    if verbose:
        print(f'Final ZX diagram has {g.num_vertices()} vertices and {g.num_edges()} edges')
    if g.num_vertices() == 0:
        return g, None
    if opt == 'greedy':
        return g, greedy_decomposition(g, verbose=verbose)
    elif opt == 'linear':
        return g, linear_decomposition(g, verbose=verbose)
    elif opt == 'dp':
        return g, dp_decomposition(g, verbose=verbose)
    elif opt == 'auto':
        return g, auto_decomposition(g, verbose=verbose)
    raise ValueError('Unknown decomposition strategy')
