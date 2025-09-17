import numpy as np
from galois import GF2
from typing import Tuple

BASE = 64


class REF:
    def __init__(self, graph):
        self.n = len(graph)
        self.N = (self.n + BASE - 1) // BASE * BASE
        self.taken = [False] * self.N
        self.graph = np.hstack([graph, np.zeros((self.n, self.N - self.n), dtype=graph.dtype)])
        self.graph = np.packbits(self.graph, axis=-1, bitorder='little').view(np.uint64)
        self.mat = []
        self.pivot_cols = []

    def _add_row(self, row_arr, start_col=0):
        pivot_row = 0
        for col in range(start_col, self.N):
            if self.taken[col]:
                continue
            while pivot_row < len(self.pivot_cols) and self.pivot_cols[pivot_row] < col:
                pivot_row += 1
            col_loc, col_rem = divmod(col, BASE)
            if (row_arr[col_loc] >> col_rem) & 1:
                if pivot_row < len(self.pivot_cols) and self.pivot_cols[pivot_row] == col:
                    row_arr ^= self.mat[pivot_row]
                else:
                    self.mat.insert(pivot_row, row_arr)
                    self.pivot_cols.insert(pivot_row, col)
                    break

    def take(self, col):
        self.taken[col] = True
        if col in self.pivot_cols:
            pivot_row = self.pivot_cols.index(col)
            self.pivot_cols.pop(pivot_row)
            row_arr = self.mat[pivot_row]
            self.mat.pop(pivot_row)
            self._add_row(row_arr, col + 1)
        self._add_row(self.graph[col])

    def rank(self):
        return len(self.pivot_cols)


def rank_factorize(A: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    assert len(A.shape) == 2 and A.dtype == np.int8
    n, m = A.shape
    A1 = A.copy()
    L, R = np.eye(n, dtype=np.int8), np.eye(m, dtype=np.int8)
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
                    A1[k] ^= A1[r]
                    L[:, r] ^= L[:, k]
            r += 1
    for i, j in list(enumerate(pivots))[::-1]:
        for k in range(i - 1, -1, -1):
            if A1[k, j]:
                A1[k] ^= A1[i]
                L[:, i] ^= L[:, k]
    for i, j in enumerate(pivots):
        for k in range(j + 1, m):
            if A1[i, k]:
                A1[:, k] ^= A1[:, j]
                R[j] ^= R[k]
    for i, j in enumerate(pivots):
        A1[:, [i, j]] = A1[:, [j, i]]
        R[[i, j]] = R[[j, i]]
    return A1.sum(), L, R


def generalized_inverse(A: np.ndarray) -> np.ndarray:
    r, U, V = rank_factorize(A)
    return (np.linalg.inv(GF2(V))[:, :r] @ np.linalg.inv(GF2(U))[:r]) == 1
