import numpy as np
from scipy.optimize import minimize
from scipy.optimize._trustregion_constr.equality_constrained_sqp import equality_constrained_sqp

from admmsolver.objectivefunc import LeastSquares, L1Regularizer, L2Regularizer
from admmsolver.optimizer import SimpleOptimizer, Model
from admmsolver.matrix import identity, DiagonalMatrix


def _randn_cmplx(*shape):
    return np.random.randn(*shape) + 1j* np.random.randn(*shape)

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
    p = Model([lstsq, l1], equality_conditions)
    opt = SimpleOptimizer(p)

    assert np.abs(opt(2*[x_ref]) - f(x_ref)) < 1e-10
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
    p = Model([lstsq, l1], equality_conditions)
    opt = SimpleOptimizer(p)

    niter = 100
    opt.solve(niter)

    np.testing.assert_allclose(opt.x[0], xanswer, atol=1e-2*np.abs(xanswer).max(), rtol=0)


def test_ridge():
    """
    |y - A * x|^2 + alpha |B x|_2,
    The solution is
       - (A^+ A + B^+B)^{-1} A^+ y
    """
    N1, N2, N3 = 2, 2, 1
    np.random.seed(100)
    y = _randn_cmplx(N1)
    A = _randn_cmplx(N1, N2)
    B = _randn_cmplx(N3, N2)
    alpha = 1

    # ADMM
    lstsq = LeastSquares(1.0, A, y)
    l2 = L2Regularizer(alpha, B)
    equality_conditions = [
       (1, 0, identity(N2), identity(N2))
    ]
    model = Model([lstsq, l2], equality_conditions)
    opt = SimpleOptimizer(model)
    opt.solve(niter=100, update_h=True)

    x_ref = np.linalg.inv(A.conjugate().T @ A + alpha * B.conjugate().T @ B) @ A.conjugate().T @ y
    np.testing.assert_allclose(opt.x[0], x_ref, atol=np.abs(x_ref).max()*1e-8)