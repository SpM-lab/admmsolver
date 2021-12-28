from admmsolver.matrix import identity, DiagonalMatrix, inv, matmul
import numpy as np

def test_diagonal_matrix():
    np.random.seed(100)

    d = np.array([1.0, 0.1])
    m = DiagonalMatrix(d)
    np.testing.assert_allclose(inv(m).diagonals, 1/d)

    a = np.random.randn(2,2)

    np.testing.assert_allclose(matmul(a, m), a * d[None,:])
    np.testing.assert_allclose(matmul(m, a), d[:,None] * a)