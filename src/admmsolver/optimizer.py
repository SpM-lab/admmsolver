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
    
    def __call__(self, x):
        """ Evaluate the cost function (without the linear constraint term) """
        return self._lstsq(x) + np.sum([p(x) for p in self._penalties])

    def solve(self, x0=None, niter=10000, nu=None, h=None, mu_p=None, callback=None):
        size_x = self._lstsq.shape[1]

        x = np.zeros((self._num_penalties+1, size_x), dtype=np.complex128)
        if x0 is not None:
            x[...] = x0[None,:]

        if mu_p is None:
            mu_p = np.ones(self._num_penalties, dtype=np.float64)

        if nu is None:
            nu = np.zeros(self._lstsq.num_constraints, dtype=np.complex128)

        if h is None:
            h = np.zeros((self._num_penalties, size_x), dtype=np.complex128)

        # Make copies because they will be updated during iterations
        mu_p = np.asarray(mu_p, dtype=np.float64)
        nu = np.array(nu, copy=True) 
        h = np.array(h, copy=True)
        
        for iter in range(niter):
            # Optimize x and nu
            h_ = np.sum(
                np.array([h[p,:] - mu_p[p] * x[p+1,:] for p in range(self._num_penalties)]),
                axis=0
            )
            mu_ = np.sum(mu_p)
            x[0, :], nu[:] = self._lstsq.solve(nu, h_, mu_)

            # Optimize x_p
            for p in range(self._num_penalties):
                h_ = - h[p,:] - mu_p[p] * x[p+1,:]
                x[p+1,:] = self._penalties[p].solve(h_, mu_p[p])

            # Optimize h_p
            for p in range(self._num_penalties):
                h[p,:] += mu_p[p] * (x[0,:] - x[p+1,:])
            
            if callback is not None:
                callback(x[0,:])

        return x # Return all x's