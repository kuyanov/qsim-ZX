import pyzx as zx
import quimb.tensor as qtn


def circuit2quimb(circ: zx.Circuit) -> qtn.Circuit:
    return qtn.Circuit.from_openqasm2_str(circ.to_qasm())


def quimb_amplitude(circ: qtn.Circuit, state: str, effect: str, **kwargs):
    circ_new = qtn.Circuit(circ.N)
    gate_map = {
        '0': ['H', 'H'],
        '1': ['X'],
        '+': ['H'],
        '-': ['X', 'H'],
        'T': ['H', 'T']
    }
    for i, ch in enumerate(state):
        circ_new.apply_gates(gate_map[ch], qubits=[i])
    circ_new.apply_gates(circ.gates)
    for i, ch in enumerate(effect):
        circ_new.apply_gates(gate_map[ch][::-1], qubits=[i])
    return circ_new.amplitude('0' * circ.N, **kwargs)
