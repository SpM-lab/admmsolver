from admmsolver.matrix import *
from admmsolver.matrix import MatrixBase
import numpy as np
from typing import Tuple

def _randn_cmplx(*shape) -> np.ndarray:
    return np.random.randn(*shape) + 1j * np.random.randn(*shape)


def test_matmal():
    np.random.seed(100)

    # (12, 12) * (12, 4)
    n1, n2, n3 = 12, 12, 4

    left_mat = []
    left_mat.append(DiagonalMatrix(np.ones(n1)))
    left_mat.append(ScaledIdentityMatrix(n1, 1+1j))
    left_mat.append(PartialDiagonalMatrix(_randn_cmplx(3,3), rest_dims=(4,)))
    left_mat.append(DenseMatrix(_randn_cmplx(n1, n2)))

    right_mat = []
    right_mat.append(DenseMatrix(_randn_cmplx(n2, n3)))
    right_mat.append(ScaledIdentityMatrix((n2, n3), 1+1j))
    right_mat.append(PartialDiagonalMatrix(_randn_cmplx(3,1), rest_dims=(4,)))

    for l in left_mat:
        for r in right_mat:
            print(type(l), type(r))
            lr = l @ r
            assert isinstance(lr, MatrixBase)
            np.testing.assert_allclose(lr.asmatrix(), l.asmatrix() @ r.asmatrix())


def test_matmal2():
    np.random.seed(100)

    # (4, 12) * (12, 12)
    n1, n2, n3 = 4, 12, 12

    left_mat = []
    left_mat.append(DenseMatrix(_randn_cmplx(n1, n2)))
    left_mat.append(PartialDiagonalMatrix(_randn_cmplx(1,3), rest_dims=(4,)))

    right_mat = []
    right_mat.append(DiagonalMatrix(np.ones(n3)))
    right_mat.append(ScaledIdentityMatrix(n3, 1+1j))
    right_mat.append(PartialDiagonalMatrix(_randn_cmplx(3,3), rest_dims=(4,)))
    right_mat.append(DenseMatrix(_randn_cmplx(n2, n3)))

    for l in left_mat:
        for r in right_mat:
            print(type(l), type(r))
            lr = l @ r
            assert isinstance(lr, MatrixBase)
            np.testing.assert_allclose(lr.asmatrix(), l.asmatrix() @ r.asmatrix())


def test_mul_transpose_conj():
    np.random.seed(100)

    # (4, 12)
    n1, n2 = 4, 12

    mat = []
    mat.append(DiagonalMatrix(np.ones(n1)))
    mat.append(DiagonalMatrix(np.ones(n1), shape=(n1, n2)))
    mat.append(ScaledIdentityMatrix(n1, 1+1j))
    mat.append(ScaledIdentityMatrix((n1,n2), 1+1j))
    mat.append(PartialDiagonalMatrix(_randn_cmplx(3,3), rest_dims=(4,)))
    mat.append(DenseMatrix(_randn_cmplx(n1, n2)))

    c = 1 + 0.1j
    for m in mat:
        print(type(m), m.shape)
        cm = c * m
        assert isinstance(cm, MatrixBase)
        np.testing.assert_allclose(cm.asmatrix(), c * m.asmatrix())

        cm = m.T
        assert isinstance(cm, MatrixBase)
        np.testing.assert_allclose(cm.asmatrix(), m.asmatrix().T)

        cc = m.conj()
        assert isinstance(cc, MatrixBase)
        np.testing.assert_allclose(cc.asmatrix(), m.asmatrix().conj())



def test_add():
    np.random.seed(100)

    n = 2

    mat = []
    mat.append(DiagonalMatrix(np.ones(n)))
    mat.append(ScaledIdentityMatrix(n, 1+1j))
    mat.append(PartialDiagonalMatrix(_randn_cmplx(n, n), (1,1)))
    mat.append(DenseMatrix(_randn_cmplx(n, n)))

    for m in mat:
        for m2 in mat:
            print(type(m), type(m2))
            m3 = m + m2
            assert isinstance(m3, MatrixBase)
            np.testing.assert_allclose(m3.asmatrix(), m.asmatrix() + m2.asmatrix())

def test_inv():
    np.random.seed(100)

    n = 4

    mat = []
    mat.append(DiagonalMatrix(np.ones(n)))
    mat.append(ScaledIdentityMatrix(n, 1+1j))
    mat.append(PartialDiagonalMatrix(_randn_cmplx(2, 2), (2,)))
    mat.append(DenseMatrix(_randn_cmplx(n, n)))

    for m in mat:
        inv_m = m.inv()
        assert isinstance(inv_m, MatrixBase)
        np.testing.assert_allclose(inv_m.asmatrix() @ m.asmatrix(), np.identity(n), rtol=0, atol=1e-12)

def test_matvec():
    np.random.seed(100)

    n = 4

    mat = []
    mat.append(DiagonalMatrix(np.ones(n)))
    mat.append(ScaledIdentityMatrix(n, 1+1j))
    mat.append(PartialDiagonalMatrix(_randn_cmplx(2, 2), (2,)))
    mat.append(DenseMatrix(_randn_cmplx(n, n)))

    vec = np.ones(n)

    for m in mat:
        print(type(m))
        mv = m @ vec
        assert isinstance(mv, np.ndarray)
        np.testing.assert_allclose(mv, m.asmatrix()@vec)