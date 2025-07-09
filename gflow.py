import pyzx as zx
from galois import GF2
from pyzx.gflow import gflow


n_qubits = 10
n_gates = 20
c = zx.generate.CNOT_HAD_PHASE_circuit(qubits=n_qubits, depth=n_gates)
g = c.to_graph()
# g.apply_state('0' * n_qubits)
# g.apply_effect('0' * n_qubits)
# zx.clifford_simp(g)
zx.full_reduce(g)
layers, nxt = gflow(g)

n = g.num_vertices()
vert = list(g.vertices())
print(vert)
mat = GF2.Zeros((n, n))
for u1, v1 in g.edge_set():
    u, v = vert.index(u1), vert.index(v1)
    mat[u][v] = mat[v][u] = 1
ord = sorted(list(layers.keys()), key=lambda v: layers[v])
print(ord)
zx.draw(g, labels=True)
