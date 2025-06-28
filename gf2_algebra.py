def gf2_rank(mat):
    assert len(mat.shape) == 2
    n, m = mat.shape
    i = 0
    for j in range(m):
        if mat[i:,j].any():
            i1 = i + mat[i:,j].argmax()
            mat[[i,i1]] = mat[[i1,i]]
            for k in range(i + 1, n):
                if mat[k,j]:
                    mat[k] ^= mat[i]
            i += 1
    return i
