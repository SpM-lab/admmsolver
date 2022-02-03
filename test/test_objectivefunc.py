import numpy as np
from scipy.optimize import minimize
from typing import cast

from admmsolver.objectivefunc import *
from admmsolver.matrix import identity, asmatrixtype, PartialDiagonalMatrix

import pytest

def _randn_cmplx(*shape) -> np.ndarray:
    return np.random.randn(*shape) + 1j* np.random.randn(*shape)

def _to_real_array(x):
    if np.iscomplexobj(x):
        return x.view(np.float64).ravel()
    else:
        return x

def _from_real_array(x):
    x = x.reshape((-1, 2))
    x = x.view(np.complex128)
    return x.reshape(x.shape[0:-1])

def _minimize(f, x0, method="BFGS"):
    x0 = _to_real_array(x0)
    res = minimize(
            lambda x: f(_from_real_array(x)),
            x0,
            method=method,
            options={'maxiter': 100000}
        )
    return _from_real_array(res.x)

def test_least_squares():
    """
    Least-squares fit with a dense coefficient matrix
    """
    np.random.seed(100)
    N1, N2 = 4, 2
    alpha = 2.0
    y = _randn_cmplx(N1)
    A = _randn_cmplx(N1, N2)
    h = _randn_cmplx(N2)
    sqrt_mu = _randn_cmplx(N2, N2)
    mu = asmatrixtype(sqrt_mu.T.conjugate() @ sqrt_mu)
    lstsq = LeastSquares(alpha, A, y)
    x = lstsq.solve(h, mu)

    f_all = lambda x: np.real(alpha * np.linalg.norm(y - A @ x)**2 + \
        h.T.conjugate() @ x + x.T.conjugate() @ h + x.conjugate().T @ (mu @ x))
    x_ref = _minimize(f_all, x, "BFGS")
    np.testing.assert_allclose(x, x_ref, rtol=1e-8)
    np.testing.assert_allclose(f_all(x), f_all(x_ref), rtol=1e-8)


def test_least_squares_partial():
    """
    Least-squares fit with a coefficient matrix of type PartialDiagonalMatrix
    """
    np.random.seed(100)
    N1, N2 = 40, 20
    alpha = 2.0
    rest_dim_size = 20
    y = _randn_cmplx(N1)
    A = PartialDiagonalMatrix(_randn_cmplx(N1//rest_dim_size, N2//rest_dim_size), rest_dims=(rest_dim_size,))
    assert A.shape == (N1, N2)
    h = _randn_cmplx(N2)
    sqrt_mu = _randn_cmplx(N2, N2)
    mu = asmatrixtype(sqrt_mu.T.conjugate() @ sqrt_mu)
    lstsq = LeastSquares(alpha, A, y)
    x = lstsq.solve(h, mu)

    f_all = lambda x: \
        np.real(
            alpha * np.linalg.norm(y - A @ x)**2 + h.T.conjugate() @ x + \
                x.T.conjugate() @ h + x.conjugate().T @ (mu @ x))
    x_ref = _minimize(f_all, x, "BFGS")
    np.testing.assert_allclose(x, x_ref, rtol=1e-4)
    np.testing.assert_allclose(f_all(x), f_all(x_ref), rtol=1e-8)


def test_constrained_least_squares():
    """
    Linearly constrained least squares
    """
    np.random.seed(100)

    N1, N2 = 8, 4
    Nc = 2
    alpha = 2.0
    y = _randn_cmplx(N1)
    A = _randn_cmplx(N1, N2)
    h = _randn_cmplx(N2)
    C = _randn_cmplx(Nc, N2)
    D = _randn_cmplx(Nc)
    sqrt_mu = _randn_cmplx(N2, N2)
    mu = asmatrixtype(sqrt_mu.T.conjugate() @ sqrt_mu)
    lstsq = ConstrainedLeastSquares(alpha, A, y, C, D)

    x = lstsq.solve(h, mu)
    assert np.abs(C@x - D).max() < 1e-10

    #FIXME: how to check x?


def test_L1():
    """
    Minimize alpha * |x|_1 + h^+ x + x^+ h + mu x^+ x
    """
    N = 20
    assert N%2 == 0
    h = 0.5*np.arange(-N//2, N//2)
    mu = identity(N)
    alpha = 1.0

    l1 = L1Regularizer(alpha, N)
    x = l1.solve(h, mu)

    # Naive optimization
    for i in range(N):
        f = lambda x: alpha * np.abs(x) + 2*h[i]*x + mu.diagonals[i] * x**2
        x0 = 0.0
        res = minimize(f, x0, method="BFGS")
        assert np.abs(x[i]-res.x[0]) < 1e-5

def test_non_negative():
    """
    Minimize infty * Theta(-x) + h^+ x + x^+ h + mu x^+ x
    """
    h = np.array([0, -10, 10])
    N = h.size
    mu = identity(N)

    func = NonNegativePenalty(N)
    x = func.solve(h, mu)

    step_f = lambda x: x if x >= 0 else 0

    # Naive optimization
    for i in range(N):
        f = lambda x: 1e+5 * step_f(-x) + 2*h[i]*x + mu.diagonals[i] * x**2
        x0 = 0.0
        res = minimize(f, x0, method="BFGS")
        assert np.abs(x[i]-res.x[0]) < 1e-5


def test_L2():
    """
    Minimize alpha * |A x|_2^2 + h^+ x + x^+ h + mu x^+ x
    """
    N = 10
    M = 5
    sqrt_mu = _randn_cmplx(N, N)
    mu = asmatrixtype(sqrt_mu.T.conjugate() @ sqrt_mu)
    alpha = 2.0
    A = _randn_cmplx(M, N)
    h = _randn_cmplx(N)

    l2 = L2Regularizer(alpha, A)
    x = l2.solve(h, mu)

    # Naive optimization
    f = lambda x: alpha * np.linalg.norm(A @ x)**2 + \
        2*np.real(h.conjugate().T @ x) + np.real(x.conjugate().T @ (mu @ x))
    x_ref = _minimize(f, x, method="BFGS")
    np.testing.assert_allclose(cast(np.ndarray, x), x_ref, atol=np.abs(x_ref).max()*1e-5, rtol=0)


def test_semi_positive_definite_penalty():
    np.random.seed(100)
    K = 20
    N = 10

    h = _randn_cmplx(N**2 * K)

    for mu in [
        asmatrixtype(identity(N**2 * K)),
        PartialDiagonalMatrix(ScaledIdentityMatrix(N**2, 1.0), (K,))]:
        p = SemiPositiveDefinitePenalty((N,N,K), axis=2)
        res = p.solve(h, mu)

        x = res.reshape((N,N,K))
        for k in range(K):
            evals, evecs = np.linalg.eigh(x[:,:,k])
            assert all(evals > -1e-10)

