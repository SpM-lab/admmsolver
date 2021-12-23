import numpy as np

from .lstsq import LeastSquaresBase

class SimpleOptimizer:
    """
    The simplest ADMM solver
    """
    def __init__(self, lstsq, penalties):
        assert isinstance(lstsq, LeastSquaresBase)
        self._num_penalties = len(penalties)
        self._lstsq = lstsq
        self._penalties = penalties

    def solve(self, x0=None, maxiter=10000, nu=None, h=None, mu=None):
        size_x = self._lstsq.shape[1]
        print("size_x", size_x)

        if mu is None:
            mu = np.ones(self._num_penalties, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)

        if nu is None:
            nu = np.ones(self._lstsq.num_constraints, dtype=np.complex128)
        nu = np.array(nu, copy=True)

        if h is None:
            h = np.ones((self._num_penalties, size_x), dtype=np.complex128)
        h = np.array(h, copy=True)
        
        x = np.zeros((self._num_penalties+1, size_x), dtype=np.complex128)
        if x0 is not None:
            x[...] = x[None,:]
        
        for iter in range(maxiter):
            # Optimize x and nu
            h_ = np.sum(
                np.array([h[p,:] - mu[p] * x[p+1,:] for p in range(self._num_penalties)]),
                axis=0
            )
            mu_ = np.sum(mu)
            x[0, :], nu[:] = self._lstsq.solve(nu, h_, mu_)

            # Optimize x_p
            for p in range(self._num_penalties):
                h_ = - h[p,:] - mu[p] * x[p+1,:]
                x[p+1,:] = self._penalties[p].solve(h_, mu[p])

            # Optimize h_p
            for p in range(self._num_penalties):
                h[p,:] += mu[p] * (x[0,:] - x[p+1,:])

            print(iter, x[0,0])

        return x # Return all x's