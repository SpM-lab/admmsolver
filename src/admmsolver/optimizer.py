# Copyright (C) 2021-2022 H. Shinaoka and others
# SPDX-License-Identifier: MIT

import numpy as np
from itertools import product
from typing import Tuple, Union, Optional, List, Sequence, cast, Callable

from .objectivefunc import ObjectiveFunctionBase
from .matrix import MatrixBase, PartialDiagonalMatrix, asmatrixtype
from .util import norm

class EqualityCondition(object):
    """
    Equality condition:
        E1 @ x_{i1} - E2 @ x_{i2} = 0,
    where i != j.
    """
    def __init__(self,
        i1: int,
        i2: int,
        E1: Union[np.ndarray, MatrixBase],
        E2: Union[np.ndarray, MatrixBase]
        ) -> None:
        assert i1 != i2, "i1 != i2!"
        assert E1.shape[0] == E2.shape[0], "Leading dimensions of E1 and E2 do not match!"
        assert E1.ndim == 2
        assert E2.ndim == 2
        E1 = asmatrixtype(E1)
        E2 = asmatrixtype(E2)
        super().__init__()
        self.i1 = i1
        self.i2 = i2
        self.E1 = E1
        self.E2 = E2

    @property
    def size(self) -> int:
        return self.E1.shape[0]

class Model(object):
    def __init__(self,
        functions: Sequence[ObjectiveFunctionBase],
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

        for ie, e in enumerate(equality_conditons):
            try:
                if isinstance(e, tuple):
                    # For backward compatibility
                    self._add_equality_condition(EqualityCondition(*e))
                else:
                    self._add_equality_condition(e)
            except Exception as e:
                print(f"Error occured when adding {ie}-th equality condition!")
                raise e

    @property
    def functions(self) -> Sequence[ObjectiveFunctionBase]:
        return self._functions

    @property
    def num_func(self) -> int:
        return self._num_func

    @property
    def E(self) -> np.ndarray:
        """ E_{ij} """
        return self._E

    def _add_equality_condition(self, e: EqualityCondition) -> None:
        """
        Add an equality condition
        """
        assert isinstance(e, EqualityCondition)
        assert e.E1.shape[1] == self._functions[e.i1].size_x, \
            f"{e.E1.shape} {self._functions[e.i1].size_x}"
        assert e.E2.shape[1] == self._functions[e.i2].size_x, \
            f"{e.E2.shape} {self._functions[e.i2].size_x}"
        if self._E[e.i1, e.i2] is not None:
            raise RuntimeError("Duplicate entries in equality_conditions")
        self._E[e.i2, e.i1] = e.E1
        self._E[e.i1, e.i2] = e.E2


# Backward compatibility
Problem = Model


class SimpleOptimizer(object):
    """
    The simplest ADMM solver
    """
    def __init__(self, model: Model, x0=None, mu=None, max_mu: float =1e+3) -> None:
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
        assert isinstance(model, Model)
        num_func = model.num_func
        self._h = np.full((num_func, num_func), None)
        self._mu = np.full((num_func, num_func), 0.0)
        self._model = model
        self._max_mu = max_mu

        if x0 is not None:
            for i in range(len(x0)):
                assert model._functions[i].size_x == x0[i].size
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
        
        self._primal_residual = [] # type: List[float]
        self._dual_residual = [] # type: List[float]

    
    @property
    def x(self) -> List[np.ndarray]:
        return self._x
    
    
    def __call__(self, x: List[np.ndarray]) -> float:
        """Evaluate the cost function"""
        return float(np.sum([f(x_) for x_, f in zip(x, self._model.functions)]))
    
    def _hk(self, k: int) -> Optional[np.ndarray]:
        """ Compute `h` for optimizing `x_k` """
        res = []

        E = self._model.E

        # i < k
        for i in range(k):
            if self._h[k,i] is None:
                continue
            res.append(
                E[i,k].T.conjugate() @ self._h[k,i]
                - self._mu[k,i] * E[i,k].T.conjugate()@ (E[k,i] @ self._x[i])
                )
            assert res[-1].size == self._model.functions[k].size_x, \
                f"{res[-1].size} {self._model.functions[k].size_x}"

        # k < i
        for i in range(k+1, self._model.num_func):
            if self._h[i,k] is None:
                continue
            res.append(
                    - E[i,k].T.conjugate() @ self._h[i,k]
                    - self._mu[i,k] * E[i,k].T.conjugate() @ (E[k,i] @ self._x[i])
                )
            assert res[-1].size == self._model.functions[k].size_x, \
                f"{res[-1].size} {self._model.functions[k].size_x}"

        if len(res) > 0:
            return _sum(res)
        else:
            return None

    def _mu_k(self, k) -> Optional[MatrixBase]:
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
 
    def check_convergence(self, rtol) -> bool:
        """ Check convergence """
        converged = True
        for i, j in product(range(self._model.num_func), repeat=2):
            if self._model.E[i,j] is None or i <= j:
                continue
            primal1 = cast(np.ndarray, self._model.E[i,j] @ self._x[j])
            primal2 = cast(np.ndarray, self._model.E[j,i] @ self._x[i])
            d_primal = cast(np.ndarray, primal1 - primal2)
            dual1 = self._mu[i,j] * self._model.E[j,i] @ (self._model.E[i,j] @ self._x[j])
            dual2 = self._mu[i,j] * self._model.E[j,i] @ (self._model.E[i,j] @ self._x_old[j])
            d_dual = cast(np.ndarray, dual1 - dual2)
            converged = converged and \
                norm(d_primal)/max(norm(primal1), norm(primal2)) < rtol
            converged = converged and \
                norm(d_dual)/max(norm(dual1), norm(dual2)) < rtol

        return converged

    def residual(self) -> Tuple[float, float]:
        """ Compute primal residual and dual residual"""
        num_func = self._model.num_func
        primal = 0.0
        dual = 0.0
        for i, j in product(range(num_func), repeat=2):
            if self._model.E[i,j] is None or i <= j:
                continue
            primal += \
                float(
                    np.linalg.norm(
                        self._model.E[i,j]@self._x[j] - self._model.E[j,i]@self._x[i]
                    )
                )
            dual += \
                float(
                    np.linalg.norm(
                        self._mu[i,j] * (
                            self._model.E[j,i] @
                            self._model.E[i,j] @ (self._x[j] - self._x_old[j])
                        )
                    )
                )
        return primal, dual


    def update_mu(self, fact_incr: float = 2.0, th_change: float = 10.0) -> None:
        """ Update mu based on primal residual and dual residual"""
        num_func = self._model.num_func
        for i, j in product(range(num_func), repeat=2):
            if self._model.E[i,j] is None or i <= j:
                continue
            primal = \
                np.linalg.norm(
                    self._model.E[i,j]@self._x[j] - 
                       self._model.E[j,i]@self._x[i]
                )
            dual = \
                np.linalg.norm(
                    self._mu[i,j] * (
                        self._model.E[j,i] @
                            self._model.E[i,j] @ (self._x[j] - self._x_old[j])
                    )
                )
            if primal > th_change * dual:
                self._mu[i,j] *= fact_incr
            if dual > th_change * primal:
                self._mu[i,j] /= fact_incr
            self._mu[i,j] = min(self._mu[i,j], self._max_mu)
        

    def solve(
            self,
            niter:int =10000,
            callback=None,
            interval_update_mu:int =100,
            update_h: bool = True,
            rtol: float = 1e-12
        ) -> None:
        for iter in range(niter):
            self.one_sweep(update_h=update_h)
            primal, dual = self.residual()
            self._primal_residual.append(primal)
            self._dual_residual.append(dual)
            if callback is not None:
                callback()
            if self.check_convergence(rtol):
                return
            if iter % interval_update_mu == 0:
                self.update_mu()

    def one_sweep(self, update_h: bool) -> None:
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
                            (self._model.E[j,i]@self._x[i]) -
                            (self._model.E[i,j]@self._x[j])
                        )


def _sum(objs):
    assert isinstance(objs, list)
    res = objs[0]
    if len(objs) > 1:
        for x in objs[1:]:
            res = res + x
    return res
