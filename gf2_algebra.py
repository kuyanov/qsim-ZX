def gf2_factorize(A):
    assert len(A.shape) == 2
    n, m = A.shape
    A0 = A.copy()
    A1 = A.copy()
    r = 0
    pivots = []
    for j in range(m):
        if A1[r:, j].any():
            pivots.append(j)
            i = r + A1[r:, j].argmax()
            A1[[r, i]] = A1[[i, r]]
            for k in range(r + 1, n):
                if A1[k, j]:
                    A1[k] ^= A1[r]
            r += 1
    for i in range(r - 1, -1, -1):
        j = pivots[i]
        for i1 in range(i - 1, -1, -1):
            if A1[i1, j]:
                A1[i1] ^= A1[i]
    return A0[:, pivots], A1[:r, :]
