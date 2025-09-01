import os
import random

import numpy as np
import pyzx as zx

from rw_simulate import simulate_circuit, circuit2graph
from quimb_helpers import circuit2quimb, quimb_amplitude


def test_simulate_special():
    circuit_dir = 'circuits/special'
    basis_states = ['0', '1', '+', '-', 'T']
    for f in sorted(os.listdir(circuit_dir)):
        fpath = os.path.join(circuit_dir, f)
        circ = zx.Circuit.load(fpath)
        state = random.choice(basis_states) * circ.qubits
        effect = random.choice(basis_states) * circ.qubits
        qcirc = circuit2quimb(circ)
        res_rw = simulate_circuit(circ, state, effect)
        res_quimb = quimb_amplitude(qcirc, state, effect)
        if not np.allclose(np.abs(res_rw), np.abs(res_quimb), atol=1e-6):
            g = circuit2graph(circ, state, effect)
            g.apply_state('0' * circ.qubits)
            g.apply_effect('0' * circ.qubits)
            corr = zx.tensorfy(g)
            rw_wrong = not np.allclose(np.abs(res_rw), np.abs(corr), atol=1e-6)
            quimb_wrong = not np.allclose(np.abs(res_quimb), np.abs(corr), atol=1e-6)
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
