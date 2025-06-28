def incidence_list(edges):
    n = max([max(u, v) for u, v, r in edges]) + 1
    inc = [[] for _ in range(n)]
    for u, v, r in edges:
        inc[u].append((v, r))
        inc[v].append((u, r))
    return inc


def root_tree(inc, root):
    children = [[] for _ in range(len(inc))]

    def dfs(u, p):
        for v, r in inc[u]:
            if v != p:
                children[u].append((v, r))
                dfs(v, u)

    dfs(root, -1)
    return children


def euler_tour(children, root):
    tour = []
    leaf_seg = [(None, None)] * len(children)

    def dfs(u):
        if not children[u]:
            tour.append(u)
            leaf_seg[u] = (len(tour) - 1, len(tour))
            return
        for v, _ in children[u]:
            dfs(v)
        first_child, last_child = children[u][0][0], children[u][-1][0]
        leaf_seg[u] = (leaf_seg[first_child][0], leaf_seg[last_child][1])

    dfs(root)
    return tour, leaf_seg


def get_partition(tour, leaf_seg, u):
    l, r = leaf_seg[u]
    return tour[l:r], tour[:l] + tour[r:]
