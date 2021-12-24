import numpy as np
from scipy.optimize import minimize

from admmsolver.lstsq import MatrixLeastSquares
from admmsolver.penalty import L1Regularizer
from admmsolver.optimizer import SimpleOptimizer

def test_LASSO():
    """
    |y - A * x|^2 + alpha |x|_1,
    where
       y = [2.]
       A = [[2., 1.]]
       alpha = 0.1
      
    The solution is x \simeq (1, 0).
    """
    y = np.array([2])
    A = np.array([[2, 1]])
    alpha = 0.1

    lstsq = MatrixLeastSquares(A, y)
    p = L1Regularizer(alpha)

    # Naive optimization
    f = lambda x: np.linalg.norm(y - A @ x)**2 + alpha * np.sum(np.abs(x))
    res = minimize(f, x0=np.array([1.1,0]), method="Nelder-Mead", options={"xatol": 1e-10})
    assert res.success 
    x_ref = res.x

    opt = SimpleOptimizer(lstsq, [p])
    assert np.abs(opt(x_ref) - f(x_ref)) < 1e-10

    opt.solve(x0=x_ref, maxiter=10)