import numpy as np

from .objectivefunc import ObjectiveFunctionBase
from .matrix import matmul

class SimpleOptimizer:
    """
    The simplest ADMM solver
    """
    def __init__(self, functions, equality_conditons=[], x0=None):
        for f in functions:
            isinstance(f, ObjectiveFunctionBase)
        self._functions = functions
        self._num_func = len(functions)
        self._equal_cond = np.full((self._num_func, self._num_func), None)
        self._h = np.full((self._num_func, self._num_func), None)
        self._E = np.full((self._num_func, self._num_func), None)
        self._mu = np.full((self._num_func, self._num_func), 0.0)

        if x0 is not None:
            self._x = [x_.copy() for x_ in x0]
        else:
            self._x = [np.zeros(self._functions[k].size_x, dtype=np.complex128)
               for k in range(self._num_func)]
        
        for e in equality_conditons:
            assert len(e) == 6
            self._add_equality_condition(*e)
    
    @property
    def x(self):
        return self._x
    
    def _add_equality_condition(self, i, j, Eji, Eij, mu, h0=None):
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
            self._h[i,j] = np.zeros(Eij.shape[0], dtype=np.complex128)
        else:
            self._h[i,j] = h0.copy()
    
    def __call__(self, x):
        """Evaluate the cost function"""
        return np.sum([f(x) for f in self._functions])
    
    def _hk(self, k):
        """ Compute `h` for optimizing `x_k` """
        res = []

        # i < k
        for i in range(k):
            if self._h[k,i] is None:
                continue
            res.append(
                matmul(self._h[k,i].T.conjugate(), self._E[i,k])
                - self._mu[k,i] * (self._E[i,k].T.conjugate()@self._E[k,i]) @ self._x[i]
                )

        # k < i
        for i in range(k+1, self._num_func):
            if self._h[i,k] is None:
                continue
            res.append(
                -matmul(self._h[i,k].T.conjugate(), self._E[i,k])
                -self._mu[i,k] *
                    matmul(
                        matmul(self._E[i,k].T.conjugate(), self._E[k,i]),
                        self._x[i]
                    )
                )
        
        if len(res) > 0:
            #return np.asarray(sum(res)).reshape((self.x[k].size,))
            return _sum(res)
        else:
            return None

    def _mu_k(self, k):
        """ Compute `mu` for optimizing `x_k` """
        res = []
        # i < k
        for i in range(k):
            if self._h[k,i] is None:
                continue
            res.append(self._mu[k,i] * self._E[i,k].T.conjugate() @ self._E[i,k])

        # k < i
        for i in range(k+1, self._num_func):
            if self._h[i,k] is None:
                continue
            res.append(self._mu[i,k] * self._E[i,k].T.conjugate() @ self._E[i,k])
        
        if len(res) > 0:
            #return np.asarray(sum(res)).reshape(2*(self.x[k].size,))
            return _sum(res)
        else:
            return None

    def solve(self, niter=10000, callback=None):
        for iter in range(niter):
            self.one_sweep()
            if callback is not None:
                callback()

    def one_sweep(self):
        """Update all variables in a single sweep"""
        # Optimize x
        for k in range(self._num_func):
            self._x[k][:] = self._functions[k].solve(
                self._hk(k),
                self._mu_k(k)
            )

        # Optimize dual variables
        for i in range(self._num_func):
            for j in range(i):
                if self._h[i,j] is not None:
                    self._h[i,j] += self._mu[i,j] * (self._x[i] - self._x[j])


def _sum(objs):
    assert isinstance(objs, list)
    res = objs[0]
    if len(objs) > 1:
        for x in objs[1:]:
           res += x
    return res