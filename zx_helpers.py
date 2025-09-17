import os
import sys
import pyzx as zx
import quizx


class capture_fd_stdout:
    """
    Context manager that captures ALL writes to the process's stdout (fd=1),
    including from Rust/C extensions using printf/println!.
    """

    def __enter__(self):
        self._saved_fd = os.dup(1)  # duplicate stdout fd
        self._r, self._w = os.pipe()  # create pipe
        os.dup2(self._w, 1)  # redirect fd=1 to pipe writer
        os.close(self._w)  # close our duplicate writer
        return self

    def __exit__(self, exc_type, exc, tb):
        os.dup2(self._saved_fd, 1)  # restore stdout
        os.close(self._saved_fd)  # close saved copy

    def read(self) -> str:
        # Read all data from the pipe reader
        chunks = []
        # Set non-blocking read in case nothing is there
        import fcntl
        flags = fcntl.fcntl(self._r, fcntl.F_GETFL)
        fcntl.fcntl(self._r, fcntl.F_SETFL, flags | os.O_NONBLOCK)
        try:
            while True:
                try:
                    data = os.read(self._r, 65536)
                    if not data:
                        break
                    chunks.append(data)
                except BlockingIOError:
                    break
        finally:
            os.close(self._r)
        return b"".join(chunks).decode(errors="replace")


def simplify_get_pivots(g: quizx.VecGraph):
    sys.stdout.flush()
    with capture_fd_stdout() as cap:
        quizx.full_simp(g)
    pivots = [tuple(map(int, l.split())) for l in cap.read().splitlines() if l.strip()]
    return pivots


def phase_gadgets(g: zx.graph.base.BaseGraph):
    gadgets = []
    for v in g.vertices():
        if g.vertex_degree(v) != 1:
            continue
        u = list(g.neighbors(v))[0]
        if v in g.inputs() or v in g.outputs() or g.phase(u) != 0:
            continue
        gadgets.append((v, u))
    return gadgets


def circuit2graph(circ: zx.Circuit, state: str = None, effect: str = None):
    new_circ = zx.Circuit(circ.qubits)
    gate_map = {
        '0': ['H', 'H'],
        '1': ['NOT'],
        '+': ['H'],
        '-': ['NOT', 'H'],
        'T': ['H', 'T']
    }
    if state:
        for i, ch in enumerate(state):
            new_circ.add_gates(' '.join(gate_map[ch]), i)
    new_circ.add_circuit(circ)
    if effect:
        for i, ch in enumerate(effect):
            new_circ.add_gates(' '.join(gate_map[ch][::-1]), i)
    g = new_circ.to_graph()
    return g
