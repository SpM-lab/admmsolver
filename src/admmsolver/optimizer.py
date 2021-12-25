import numpy as np

from .objectivefunc import ObjectiveFunctionBase

class SimpleOptimizer:
    """
    The simplest ADMM solver
    """
    def __init__(self, functions, equality_conditons=[]):
        for f in functions:
            isinstance(f, ObjectiveFunctionBase)
        self._functions = functions
        self._num_func = len(functions)
        self._equal_cond = np.full((self._num_func, self._num_func), None)
        self._h = np.full((self._num_func, self._num_func), None)
        self._mu = np.full((self._num_func, self._num_func), 0.0)
    
    def add_equality_condition(self, i, j, Eji, Eij, mu, h0=None):
        """
        Add an equality condition

        i > j
        """
        assert i > j, "i <= j!"
        assert mu > 0, "mu must be positive!"
        if self._equal_cond[i,j] is not None:
            raise RuntimeError("Duplicate entries in equality_conditions")

        self._E[i,j] = Eij
        self._E[j,i] = Eji
        self._mu[i,j] = mu
        if h0 is None:
            h0 = np.zeros(Eij.shape[0], dtype=np.complex128)
        else:
            self._h[i,j] = h0.copy()
    
    def __call__(self, x):
        """Evaluate the cost function"""
        return np.sum([f(x) for f in self._functions])
    
    def _h(self, k):
        """ Compute `h` for optimize `x_k` """
        res = []

        for i in range(k):
            if self._h[k,i] is None:
                continue
            res.append(
                self._h[k,i].T.conjugate() @ self._E[i,k]
                - self._mu[k,i] @ (self._E[i,k].T.conjugate()@self._E[k,i]) @ self._x[i]
                )

        for i in range(k+1, self._num_func):
            if self._h[i,k] is None:
                continue
            res.append(
                -self._h[i,k].T.conjugate() @ self._E[i,k]
                -self._mu[i,k] @ (self._E[i,k].T.conjugate()@self._E[k,i]) @ self._x[i]
                )
        
        return np.sum(np.array(res), axis=0)



    def solve(self, x0=None, niter=10000, callback=None):
        size_x = self._lstsq.shape[1]

        x = np.zeros((self._num_penalties+1, size_x), dtype=np.complex128)
        if x0 is not None:
            x[...] = x0[None,:]

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