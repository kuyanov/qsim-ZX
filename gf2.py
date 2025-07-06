import numpy as np
from galois import GF2


def rank_factorize(A: GF2) -> (int, GF2, GF2):
    assert len(A.shape) == 2
    n, m = A.shape
    A0, A1 = A.copy(), A.copy()
    L, R = GF2.Identity(n), GF2.Identity(m)
    pivots = []
    r = 0
    for j in range(m):
        if A1[r:, j].any():
            pivots.append(j)
            i = r + np.where(A1[r:, j])[0][0]
            A1[[r, i]] = A1[[i, r]]
            L[:, [r, i]] = L[:, [i, r]]
            for k in range(r + 1, n):
                if A1[k, j]:
                    A1[k] += A1[r]
                    L[:, r] += L[:, k]
            r += 1
    for i, j in list(enumerate(pivots))[::-1]:
        for k in range(i - 1, -1, -1):
            if A1[k, j]:
                A1[k] += A1[i]
                L[:, i] += L[:, k]
    for i, j in enumerate(pivots):
        for k in range(j + 1, m):
            if A1[i, k]:
                A1[:, k] += A1[:, j]
                R[j] += R[k]
    for i, j in enumerate(pivots):
        A1[:, [i, j]] = A1[:, [j, i]]
        R[[i, j]] = R[[j, i]]
    return int((A1 == 1).sum()), L, R


def generalized_inverse(A: GF2) -> GF2:
    r, U, V = rank_factorize(A)
    return np.linalg.inv(V)[:, :r] @ np.linalg.inv(U)[:r]
