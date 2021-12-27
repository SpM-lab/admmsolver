import numpy as np
from scipy.optimize import minimize
from scipy.optimize._trustregion_constr.equality_constrained_sqp import equality_constrained_sqp

from admmsolver.objectivefunc import LeastSquares, L1Regularizer
from admmsolver.optimizer import SimpleOptimizer, Problem
from admmsolver.matrix import identity

def test_LASSO():
    """
    |y - A * x|^2 + alpha |x|_1,
    where
       y = [2.]
       A = [[2., 1.]]
       alpha = 0.1
      
    The solution is x simeq (1, 0).

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
       (1, 0, identity(2), identity(2))
    ]
    p = Problem([lstsq, l1], equality_conditions)
    print("debug", p.E)
    opt = SimpleOptimizer(p)

    assert np.abs(opt(x_ref) - f(x_ref)) < 1e-10
    opt.solve(100)
    x_res = opt.x
    for x in x_res:
        np.testing.assert_allclose(x, x_ref, atol=1e-10)

def test_basis_pursuit():
    # Fig. 3 of J. Otsuki et al., JPSJ89, 012001 (2020)
    # https://github.com/SpM-lab/CS-tools/tree/master/jpsj-review/basis_pursuit
    # Dimension of the signal
    N = 1000

    M = 100
    K = 20    
    seed = 1234
    np.random.seed(seed)
    A = np.random.randn(M,N)
   
    #Make answer vector
    xanswer = np.zeros(N)
    xanswer[:K] = np.random.randn(K)
    xanswer = np.random.permutation(xanswer)
   
    y_calc = np.dot(A, xanswer)    

    lstsq = LeastSquares(1.0, A, y_calc)
    l1 = L1Regularizer(1e-1, A.shape[1])
    equality_conditions = [
      (1, 0, identity(N), identity(N))
    ]
    p = Problem([lstsq, l1], equality_conditions)
    opt = SimpleOptimizer(p)

    niter = 100
    opt.solve(niter)

    np.testing.assert_allclose(opt.x[0], xanswer, atol=1e-2*np.abs(xanswer).max(), rtol=0)