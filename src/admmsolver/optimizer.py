import numpy as np

from .objectivefunc import ObjectiveFunctionBase
from .matrix import matmul
from itertools import product

class Problem(object):
    def __init__(self, functions, equality_conditons=[]):
        """
        functions: list of instances of subclasses of ObjectiveFunctionBase
            Define the cost function as the sum of the given functions.

        equality_conditions: list of tuple of (int, int, 2d array, 2d array)
            Each entry denotes (i, j, Eji, Eij).
            The equality condition is E_ji x_i = E_ji x_j.
        """
        for f in functions:
            isinstance(f, ObjectiveFunctionBase)
        self._functions = functions
        self._num_func = len(functions)
        self._E = np.full((self._num_func, self._num_func), None)

        for e in equality_conditons:
            assert len(e) == 4
            self._add_equality_condition(*e)

    @property
    def functions(self):
        return self._functions
    
    @property
    def num_func(self):
        return self._num_func
    
    @property
    def E(self):
        """ E_{ij} """
        return self._E

    def _add_equality_condition(self, i, j, Eji, Eij):
        """
        Add an equality condition
        E_{ij} x_i = E_{ji} x_j
        """
        assert i != j, "i != j!"
        if self._E[i,j] is not None:
            raise RuntimeError("Duplicate entries in equality_conditions")
        self._E[i,j] = Eij
        self._E[j,i] = Eji


class SimpleOptimizer(object):
    """
    The simplest ADMM solver
    """
    def __init__(self, problem, x0=None, mu=None):
        """
        problem: Problem instance
           Problem to be solved.

        x0: None or list of 1D array
           Initial guesses for variables `x_i`
        
        mu: float
           Penalty term
        """
        num_func = problem.num_func
        self._h = np.full((num_func, num_func), None)
        self._mu = np.full((num_func, num_func), 0.0)
        self._problem = problem

        if x0 is not None:
            self._x = [x_.copy() for x_ in x0]
        else:
            self._x = [np.zeros(problem.functions[k].size_x, dtype=np.complex128)
               for k in range(num_func)]
        
        if mu is None:
            mu = 1.0
        for i, j in product(range(num_func), repeat=2):
            if problem.E[i,j] is None or i <= j:
                continue
            self._h[i,j] = np.zeros(problem.E[i,j].shape[0], dtype=np.complex128)
            self._mu[i,j] = mu

    
    @property
    def x(self):
        return self._x
    
    
    def __call__(self, x):
        """Evaluate the cost function"""
        return np.sum([f(x) for f in self._problem.functions])
    
    def _hk(self, k):
        """ Compute `h` for optimizing `x_k` """
        res = []

        E = self._problem.E

        # i < k
        for i in range(k):
            if self._h[k,i] is None:
                continue
            res.append(
                matmul(self._h[k,i].T.conjugate(), E[i,k])
                - self._mu[k,i] * (E[i,k].T.conjugate()@E[k,i]) @ self._x[i]
                )

        # k < i
        for i in range(k+1, self._problem.num_func):
            if self._h[i,k] is None:
                continue
            res.append(
                -matmul(self._h[i,k].T.conjugate(), E[i,k])
                -self._mu[i,k] *
                    matmul(
                        matmul(E[i,k].T.conjugate(), E[k,i]),
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

        E = self._problem.E

        res = []
        # i < k
        for i in range(k):
            if self._h[k,i] is None:
                continue
            res.append(self._mu[k,i] * E[i,k].T.conjugate() @ E[i,k])

        # k < i
        for i in range(k+1, self._problem.num_func):
            if self._h[i,k] is None:
                continue
            res.append(self._mu[i,k] * E[i,k].T.conjugate() @ E[i,k])
        
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
        for k in range(self._problem.num_func):
            self._x[k][:] = self._problem.functions[k].solve(
                self._hk(k),
                self._mu_k(k)
            )

        # Optimize dual variables
        for i in range(self._problem.num_func):
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