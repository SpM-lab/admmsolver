import numpy as np

from admmsolver.least_squares import *

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
        x = lstsq.solve_for_x(nu, h, mu)
        nu = lstsq.solve_for_nu(y, nu, h, mu)
    
    np.testing.assert_allclose(x, np.ones(1))


def test_2x2_least_squares():
    # Solution should be x = 1
    # y: [1]
    # A: [1, 1]
    # V: [1, -1]
    # The solution is x = [1, 0].
    A = np.array([[1,1]], dtype=np.complex128)
    y = np.array([1], dtype=np.complex128)
    V = np.array([[1, +1]], dtype=np.complex128)
    lstsq = MatrixLeastSquares(A, y, V)

    A2 = A.conjugate().T @ A
    print(np.linalg.eigh(A2))

    # Initial guess
    x = np.zeros(2, dtype=np.complex128)
    nu = np.zeros(1, dtype=np.complex128)

    # No external fields
    mu = 1e-8 # To avoid the singular matrix (A^+ A)^{-1}
    h = np.zeros_like(x)

    for _ in range(4):
        print("x", x)
        x = lstsq.solve_for_x(nu, h, mu)
        nu = lstsq.solve_for_nu(y, nu, h, mu)
    
    #np.testing.assert_allclose(x, np.ones(1))

