import numpy as np


def incidence_list(edges):
    n = len(edges) // 2 + 1
    inc = [[] for _ in range(n)]
    for e, (_, v, _) in enumerate(edges):
        inc[v].append(e)
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
