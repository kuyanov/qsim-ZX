import numpy as np


def gf2_factorize(A: np.ndarray) -> (np.ndarray, np.ndarray):
    assert len(A.shape) == 2
    n, m = A.shape
    A0, A1 = A.copy(), A.copy()
    pivots = []
    r = 0
    for j in range(m):
        if A1[r:, j].any():
            pivots.append(j)
            i = r + np.where(A1[r:, j])[0][0]
            A1[[r, i]] = A1[[i, r]]
            for k in range(r + 1, n):
                if A1[k, j]:
                    A1[k] ^= A1[r]
            r += 1
    for i in range(r - 1, -1, -1):
        j = pivots[i]
        for k in range(i - 1, -1, -1):
            if A1[k, j]:
                A1[k] ^= A1[i]
    return A0[:, pivots], A1[:r, :]
