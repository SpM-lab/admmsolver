import numpy as np
from scipy.optimize import minimize

from admmsolver.penalty import *

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