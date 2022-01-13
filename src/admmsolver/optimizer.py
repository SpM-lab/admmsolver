import numpy as np

from .objectivefunc import ObjectiveFunctionBase
from .matrix import matmul, DiagonalMatrix
from itertools import product
from typing import Union, Optional, List


class EqualityCondition(object):
    """
    Equality condition:
       E1 @ x_{i1} - E2 @ x_{i2} = 0,
    where i != j.
    """
    def __init__(self,
        i1: int,
        i2: int,
        E1: Union[np.ndarray, DiagonalMatrix],
        E2: Union[np.ndarray, DiagonalMatrix]
        ) -> None:
        assert i1 != i2, "i1 != i2!"
        assert E1.shape[0] == E2.shape[0], "Leading dimensions of E1 and E2 do not match!"
        assert E1.ndim == 2
        assert E2.ndim == 2
        super().__init__()
        self.i1 = i1
        self.i2 = i2
        self.E1 = E1
        self.E2 = E2
    
    @property
    def size(self):
        return self.E1.shape[0]

class Model(object):
    def __init__(self,
        functions: ObjectiveFunctionBase,
        equality_conditons: Union[tuple, List[EqualityCondition]] =[]
        ) -> None:
        """
        functions: list of instances of subclasses of ObjectiveFunctionBase
            Define the cost function as the sum of the given functions.

        equality_conditions:
            Equality conditions. Using tuples are deprecated.
        """
        for f in functions:
            isinstance(f, ObjectiveFunctionBase)
        self._functions = functions
        self._num_func = len(functions)
        self._E = np.full((self._num_func, self._num_func), None)

        # For backward compatibility
        for idx_eq in range(len(equality_conditons)):
            if isinstance(equality_conditons[idx_eq], tuple):
                equality_conditons[idx_eq] = EqualityCondition(*equality_conditons[idx_eq])

        for e in equality_conditons:
            self._add_equality_condition(e)

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

    def _add_equality_condition(self, e: EqualityCondition):
        """
        Add an equality condition
        """
        assert e.E1.shape[1] == self._functions[e.i1].size_x
        assert e.E2.shape[1] == self._functions[e.i2].size_x
        if self._E[e.i1, e.i2] is not None:
            raise RuntimeError("Duplicate entries in equality_conditions")
        self._E[e.i1, e.i2] = e.E1
        self._E[e.i2, e.i1] = e.E2


# Backward compatibility
Problem = Model


class SimpleOptimizer(object):
    """
    The simplest ADMM solver
    """
    def __init__(self, model: Model, x0=None, mu=None, max_mu: float =1e+3):
        """
        model:
           Model to be solved.

        x0: None or list of 1D array
           Initial guesses for variables `x_i`
        
        mu: float
           Penalty term

        max_mu:
           Max value of mu
        """
        num_func = model.num_func
        self._h = np.full((num_func, num_func), None)
        self._mu = np.full((num_func, num_func), 0.0)
        self._model = model
        self._max_mu = max_mu

        if x0 is not None:
            self._x = [x_.copy() for x_ in x0]
        else:
            self._x = [np.zeros(model.functions[k].size_x, dtype=np.complex128)
               for k in range(num_func)]
        
        if mu is None:
            mu = 1.0
        for i, j in product(range(num_func), repeat=2):
            if model.E[i,j] is None or i <= j:
                continue
            self._h[i,j] = np.zeros(model.E[i,j].shape[0], dtype=np.complex128)
            self._mu[i,j] = mu
        
        self._primal_residual = []
        self._dual_residual = []

    
    @property
    def x(self):
        return self._x
    
    
    def __call__(self, x):
        """Evaluate the cost function"""
        return np.sum([f(x) for f in self._model.functions])
    
    def _hk(self, k):
        """ Compute `h` for optimizing `x_k` """
        res = []

        E = self._model.E

        # i < k
        for i in range(k):
            if self._h[k,i] is None:
                continue
            res.append(
                matmul(E[i,k].T.conjugate(), self._h[k,i])
                - self._mu[k,i] * (E[i,k].T.conjugate()@E[k,i]) @ self._x[i]
                )

        # k < i
        for i in range(k+1, self._model.num_func):
            if self._h[i,k] is None:
                continue
            res.append(
                -matmul(E[i,k].T.conjugate(), self._h[i,k])
                -self._mu[i,k] *
                    matmul(
                        matmul(E[i,k].T.conjugate(), E[k,i]),
                        self._x[i]
                    )
                )
        
        if len(res) > 0:
            return _sum(res)
        else:
            return None

    def _mu_k(self, k):
        """ Compute `mu` for optimizing `x_k` """

        E = self._model.E

        res = []
        # i < k
        for i in range(k):
            if self._h[k,i] is None:
                continue
            res.append(self._mu[k,i] * E[i,k].T.conjugate() @ E[i,k])

        # k < i
        for i in range(k+1, self._model.num_func):
            if self._h[i,k] is None:
                continue
            res.append(self._mu[i,k] * E[i,k].T.conjugate() @ E[i,k])
        
        if len(res) > 0:
            return _sum(res)
        else:
            return None
    
    def residual(self):
        """ Compute primal residual and dual residual"""
        num_func = self._model.num_func
        primal = 0.0
        dual = 0.0
        for i, j in product(range(num_func), repeat=2):
            if self._model.E[i,j] is None or i <= j:
                continue
            primal += \
                np.linalg.norm(
                    matmul(self._model.E[i,j], self._x[j]) - 
                       matmul(self._model.E[j,i], self._x[i])
                )
            dual += \
                np.linalg.norm(
                    self._mu[i,j] * matmul(
                        self._model.E[j,i],
                        matmul(
                            self._model.E[i,j],
                            self._x[j] - self._x_old[j]
                        )
                    )
                )
        return primal, dual


    def update_mu(self, fact_incr=2, th_change=10):
        """ Update mu based on primal residual and dual residual"""
        num_func = self._model.num_func
        for i, j in product(range(num_func), repeat=2):
            if self._model.E[i,j] is None or i <= j:
                continue
            primal = \
                np.linalg.norm(
                    matmul(self._model.E[i,j], self._x[j]) - 
                       matmul(self._model.E[j,i], self._x[i])
                )
            dual = \
                np.linalg.norm(
                    self._mu[i,j] * matmul(
                        self._model.E[j,i],
                        matmul(
                            self._model.E[i,j],
                            self._x[j] - self._x_old[j]
                        )
                    )
                )
            if primal > th_change * dual:
                self._mu[i,j] *= fact_incr
            if dual > th_change * primal:
                self._mu[i,j] /= fact_incr
            self._mu[i,j] = min(self._mu[i,j], self._max_mu)
            print(self._mu[i,j])
        

    def solve(
            self,
            niter:int =10000,
            callback=None,
            interval_update_mu:int =100,
            update_h: bool = True
        ):
        for iter in range(niter):
            self.one_sweep(update_h=update_h)
            primal, dual = self.residual()
            self._primal_residual.append(primal)
            self._dual_residual.append(dual)
            if callback is not None:
                callback()
            if iter % interval_update_mu == 0:
                self.update_mu()

    def one_sweep(self, update_h: bool):
        """Update all variables in a single sweep"""
        self._x_old = [x_.copy() for x_ in self._x]

        # Optimize x
        for k in range(self._model.num_func):
            self._x[k][:] = self._model.functions[k].solve(
                self._hk(k),
                self._mu_k(k)
            )

        # Optimize dual variables
        if update_h:
            for i in range(self._model.num_func):
                for j in range(i):
                    if self._h[i,j] is not None:
                        self._h[i,j] += self._mu[i,j] * (
                            matmul(self._model.E[j,i], self._x[i]) -
                            matmul(self._model.E[i,j], self._x[j])
                        )


def _sum(objs):
    assert isinstance(objs, list)
    res = objs[0]
    if len(objs) > 1:
        for x in objs[1:]:
           res += x
    return res
