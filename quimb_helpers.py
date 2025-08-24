import pyzx as zx
import quimb.tensor as qtn


def circuit2quimb(circ: zx.Circuit) -> qtn.Circuit:
    return qtn.Circuit.from_openqasm2_str(circ.to_qasm())


def amplitude(circ: qtn.Circuit, state: str, effect: str, **kwargs):
    circ_new = qtn.Circuit(circ.N)

    for i, ch in enumerate(state):
        if ch == '0':
            continue
        elif ch == '1':
            circ_new.apply_gate('X', i)
        elif ch == '+':
            circ_new.apply_gate('H', i)
        elif ch == '-':
            circ_new.apply_gate('X', i)
            circ_new.apply_gate('H', i)
        else:
            raise ValueError(f"Unsupported symbol {ch!r}")

    circ_new.apply_gates(circ.gates)

    for i, ch in enumerate(effect):
        if ch == '0':
            continue
        elif ch == '1':
            circ_new.apply_gate('X', i)
        elif ch == '+':
            circ_new.apply_gate('H', i)
        elif ch == '-':
            circ_new.apply_gate('H', i)
            circ_new.apply_gate('X', i)
        else:
            raise ValueError(f"Unsupported symbol {ch!r}")

    return circ_new.amplitude('0' * circ.N, **kwargs)
