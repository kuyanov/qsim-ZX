import os

import numpy as np
import pyzx as zx

from rw_simulate import simulate_circuit
from quimb_helpers import circuit2quimb, amplitude


def test_simulate_special():
    circuit_dir = 'circuits/all'
    for f in sorted(os.listdir(circuit_dir)):
        fpath = os.path.join(circuit_dir, f)
        circ = zx.Circuit.load(fpath)
        state, effect = '-' * circ.qubits, '-' * circ.qubits
        qcirc = circuit2quimb(circ)
        res_rw = simulate_circuit(circ, state, effect)
        res_quimb = amplitude(qcirc, state, effect, optimize='auto')
        if not np.allclose(np.abs(res_rw), np.abs(res_quimb)):
            g = circ.to_graph()
            g.apply_state(state)
            g.apply_effect(effect)
            corr = zx.tensorfy(g)
            rw_wrong = not np.allclose(res_rw, corr)
            quimb_wrong = not np.allclose(res_quimb, corr)
            print(f)
            if rw_wrong and not quimb_wrong:
                print(f'Rank-width strategy is wrong, expected: {corr}, found: {res_rw}')
            elif quimb_wrong and not rw_wrong:
                print(f'Quimb is wrong, expected: {corr}, found: {res_quimb}')
            elif rw_wrong and quimb_wrong:
                print('Both rank-width strategy and Quimb are wrong')
                print(f'Rank-width result: {res_rw}')
                print(f'Quimb result: {res_quimb}')
                print(f'Correct amplitude: {corr}')
            assert False
