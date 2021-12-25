import numpy as np
from scipy.optimize import minimize

from admmsolver.objectivefunc import *

def _randn_cmplx(*shape):
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
    res = minimize(lambda x: f(_from_real_array(x)), x0, method=method)
    assert res.success
    return _from_real_array(res.x)


def test_least_squares():
    np.random.seed(100)

    N1, N2 = 4, 2
    alpha = 2.0
    y = _randn_cmplx(N1)
    A = _randn_cmplx(N1, N2)
    h = _randn_cmplx(N2)
    sqrt_mu = _randn_cmplx(N2, N2)
    mu = sqrt_mu.T.conjugate() @ sqrt_mu
    lstsq = LeastSquares(alpha, A, y)

    x = lstsq.solve(h, mu)

    f = lambda x: np.real(alpha * np.linalg.norm(y - A @ x)**2)
    f_all = lambda x: np.real(alpha * np.linalg.norm(y - A @ x)**2 + h.T.conjugate() @ x + x.T.conjugate() @ h + x.conjugate().T @ mu @ x)

    x_ref = _minimize(f_all, x, "BFGS")

    np.testing.assert_allclose(x, x_ref, rtol=1e-8)
    np.testing.assert_allclose(f_all(x), f_all(x_ref), rtol=1e-8)