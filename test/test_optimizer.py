import numpy as np

from admmsolver.lstsq import MatrixLeastSquares
from admmsolver.penalty import L1Regularizer
from admmsolver.optimizer import SimpleOptimizer

def test_LASSO():
    """
    |y - A * x| + alpha |x|_1,
    where
       y = [2.]
       A = [[2., 1.]]
       alpha = 0.1
    """
    y = np.array([2])
    A = np.array([[2, 1]])
    alpha = 0.01

    lstsq = MatrixLeastSquares(A, y)
    p = L1Regularizer(alpha)
    opt = SimpleOptimizer(lstsq, [p])
    opt.solve(maxiter=200)