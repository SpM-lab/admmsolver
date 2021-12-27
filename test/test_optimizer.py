import numpy as np
from scipy.optimize import minimize
from scipy.optimize._trustregion_constr.equality_constrained_sqp import equality_constrained_sqp

from admmsolver.objectivefunc import LeastSquares, L1Regularizer
from admmsolver.optimizer import SimpleOptimizer
from admmsolver.matrix import identity

def test_LASSO():
    """
    |y - A * x|^2 + alpha |x|_1,
    where
       y = [2.]
       A = [[2., 1.]]
       alpha = 0.1
      
    The solution is x \simeq (1, 0).

    This problem can be solved by minimizing
       |y - A * x0|^2 + alpha |x1|_1
    subject to x0 - x1 = 0.
    """
    y = np.array([2])
    A = np.array([[2, 1]])
    alpha = 0.1

    # Naive optimization
    f = lambda x: np.linalg.norm(y - A @ x)**2 + alpha * np.sum(np.abs(x))
    res = minimize(f, x0=np.array([1.1,0]), method="Nelder-Mead", options={"xatol": 1e-10})
    assert res.success 
    x_ref = res.x

    # ADMM
    lstsq = LeastSquares(1.0, A, y)
    l1 = L1Regularizer(alpha, A.shape[1])
    equality_conditions = [
       (1, 0, identity(2), identity(2), 1.0, None)
    ]
    opt = SimpleOptimizer([lstsq, l1], equality_conditions)

    assert np.abs(opt(x_ref) - f(x_ref)) < 1e-10
    opt.solve(100)
    x_res = opt.x
    for x in x_res:
        np.testing.assert_allclose(x, x_ref, atol=1e-10)

    