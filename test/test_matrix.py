from admmsolver.matrix import identity, DiagonalMatrix, inv, matmul, PartialDiagonalMatrix
import numpy as np

def _randn_cmplx(*shape):
    return np.random.randn(*shape) + 1j* np.random.randn(*shape)

def test_diagonal_matrix():
    np.random.seed(100)

    d = np.array([1.0, 0.1])
    m = DiagonalMatrix(d)
    np.testing.assert_allclose(inv(m).diagonals, 1/d)

    a = np.random.randn(2,2)

    np.testing.assert_allclose(matmul(a, m), a * d[None,:])
    np.testing.assert_allclose(matmul(m, a), d[:,None] * a)

def test_partial_diagonal_matrix():
    np.random.seed(100)
    n1, n2 = 3, 2
    rest_dims = (4,5)
    a = _randn_cmplx(n1, n2)
    pdm = PartialDiagonalMatrix(a, rest_dims)

    x = _randn_cmplx(n2, *rest_dims)
    np.testing.assert_allclose(
        pdm.matvec(x.ravel()),
        np.einsum('ij,jkl->ikl', a, x).ravel()
    )

    y = np.random.randn(n1, *rest_dims)
    np.testing.assert_allclose(
        pdm.rmatvec(y.ravel()),
        np.einsum('ij,ikl->jkl', a.conj(), y).ravel()
    )

    pdm2 = pdm.conjugate().T @ pdm
    x = _randn_cmplx(n2, *rest_dims)
    np.testing.assert_allclose(
        pdm2 @ x.ravel(),
        np.einsum('ij,jkl->ikl', a.conj().T @ a, x).ravel()
    )



def test_partial_diagonal_matrix_matvec():
    np.random.seed(100)
    n1, n2 = 3, 2
    rest_dims = (4,5)
    a = _randn_cmplx(n1, n2)
    pdm = PartialDiagonalMatrix(a, rest_dims)

    n3 = 4
    x = _randn_cmplx(n2, *rest_dims, n3)
    np.testing.assert_allclose(
        pdm.matvec(x.ravel()),
        np.einsum('ij,jklR->iklR', a, x).ravel()
    )
