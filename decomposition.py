
# def sub_decomposition(decomp, new_leaves: Sequence[int]):
#     def iterate(elem):
#         if isinstance(elem, int):
#             return elem if elem in new_leaves else None
#         child1, child2 = iterate(elem[0]), iterate(elem[1])
#         if child1 is None:
#             return child2
#         if child2 is None:
#             return child1
#         return [child1, child2]
#
#     return iterate(decomp)


# def pauli_flow_decomposition(g: zx.graph.base.BaseGraph):
#     layers = gflow(g, pauli=True)[0]
#     order = (list(g.inputs()) +
#              sorted(list(layers.keys()), key=lambda v: layers[v]) +
#              list(g.outputs()))
#     order_pos = {v: i for i, v in enumerate(order)}
#
#     v_groups = []
#     for v in g.vertices():
#         if g.vertex_degree(v) != 1:
#             continue
#         u = list(g.neighbors(v))[0]
#         if v in g.inputs() or v in g.outputs() or g.phase(u) != 0:
#             continue
#         v_groups.append([v, u])
#     v_single = set(g.vertices()) - set(sum(v_groups, []))
#     v_groups += [[v] for v in v_single]
#     v_groups.sort(key=lambda v_group: max(order_pos[v] for v in v_group))
#
#     elems = [v_group[0] if len(v_group) == 1 else v_group for v_group in v_groups]
#     decomp = elems[0]
#     for elem in elems[1:]:
#         decomp = [decomp, elem]
#     return decomp
