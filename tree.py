import numpy as np

from graph import cut_rank


def incidence_list(edges):
    n = len(edges) // 2 + 1
    inc = [[] for _ in range(n)]
    for e, edge in enumerate(edges):
        inc[edge[1]].append(e)
    return inc


def calc_partitions(inc, edges):
    n, m = len(inc) // 2 + 1, len(edges)
    part = np.zeros((m, n), dtype=np.bool_)

    def dfs(e):
        if part[e].any():
            return
        u = edges[e][0]
        if len(inc[u]) == 1:
            part[e][u] = True
            return
        for f in inc[u]:
            if e ^ f != 1:
                dfs(f)
                part[e] |= part[f]

    for e in range(m):
        dfs(e)

    return part


def calc_ranks(mat, part, edges):
    new_edges = []
    for e, (u, v) in enumerate(edges):
        r = cut_rank(mat, part[e])
        new_edges.append((u, v, r))
    return new_edges


def induced_tree(inc, edges, leaves):
    new_edges = []

    def dfs(e):
        u = edges[e][0]
        if len(inc[u]) == 1:
            return leaves.index(u) if u in leaves else -1
        children = []
        for f in inc[u]:
            if e ^ f != 1:
                v = dfs(f)
                if v != -1:
                    children.append(v)
        if len(children) == 2:
            root = len(leaves) + len(new_edges) // 4
            new_edges.append((root, children[0]))
            new_edges.append((children[0], root))
            new_edges.append((root, children[1]))
            new_edges.append((children[1], root))
            return root
        elif len(children) == 1:
            return children[0]
        else:
            return -1

    def remove_vertex(v):
        nb = [e[1] for e in new_edges if e[0] == v]
        for u in nb:
            new_edges.pop(new_edges.index((v, u)))
            new_edges.pop(new_edges.index((u, v)))
        if len(nb) == 2:
            new_edges.append((nb[0], nb[1]))
            new_edges.append((nb[1], nb[0]))

    u = dfs(0)
    v = dfs(1)
    if u == -1 and v != -1:
        remove_vertex(v)
    elif u != -1 and v == -1:
        remove_vertex(u)
    elif u != -1 and v != -1:
        new_edges.append((u, v))
        new_edges.append((v, u))
    return new_edges
