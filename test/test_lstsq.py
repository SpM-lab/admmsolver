import numpy as np
from scipy.optimize import minimize

from admmsolver.objectivefunc import *

def _randn_cmplx(*shape):
    return np.random.randn(*shape) + 1j* np.random.randn(*shape)

def test_least_squares():
    # Solution should be x = 1
    np.random.seed(100)

    N1, N2 = 10, 4
    y = _randn_cmplx(N1)
    A = _randn_cmplx(N1, N2)
    h = _randn_cmplx(N2)
    sqrt_mu = _randn_cmplx(N1, N1)
    mu = sqrt_mu.T.conjugate() @ sqrt_mu
    lstsq = LeastSquares(2.0, A, y)

    
    np.testing.assert_allclose(x, np.ones(1))


def test_underdetermined_least_squares():
    # Solution should be x = 1
    # y: [1]
    # A: [1, 1]
    # V: [1, -1]
    # The solution is x = [1, 0].
    print()
    A = np.array([[1,1]], dtype=np.complex128)
    y = np.array([1], dtype=np.complex128)
    V = np.array([[1, -1]], dtype=np.complex128)
    lstsq = MatrixLeastSquares(A, y, V)

    A2 = A.conjugate().T @ A

    # Initial guess
    x = np.array([1,0], dtype=np.complex128)
    nu = np.full(1, 10.0)

    # No external fields
    mu = 1e-8 # To avoid the singular matrix (A^+ A)^{-1}
    h = np.zeros_like(x)

    for _ in range(10):
        x, nu = lstsq.solve(nu, h, mu)

    np.testing.assert_allclose(x, np.array([1, 0]), atol=1e-6)


def test_L1():
    """
    Minimize alpha * |x|_1 + h^+ x + x^+ h + mu x^+ x
    """
    N = 20
    assert N%2 == 0
    h = 0.5*np.arange(-N//2, N//2)
    mu = 1.0
    alpha = 1.0

    l1 = L1Regularizer(alpha)
    x = l1.solve(h, mu)
    
    # Naive optimization
    for i in range(N):
        f = lambda x: alpha * np.abs(x) + 2*h[i]*x + mu * x**2
        x0 = 0.0
        res = minimize(f, x0, method="BFGS")
        assert np.abs(x[i]-res.x[0]) < 1e-5