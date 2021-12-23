import numpy as np

from admmsolver.lstsq import *

def test_simplest_least_squares():
    # Solution should be x = 1
    A = np.array([[1]], dtype=np.complex128)
    y = np.array([1], dtype=np.complex128)
    V = np.array([[1]], dtype=np.complex128)
    lstsq = MatrixLeastSquares(A, y, V)

    # Initial guess
    x = np.zeros(1, dtype=np.complex128)
    nu = np.zeros(1, dtype=np.complex128)

    # No external fields
    mu = 0.0
    h = np.zeros(1)

    for _ in range(2):
        x, nu = lstsq.solve(nu, h, mu)
    
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

