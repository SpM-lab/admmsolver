# Copyright (C) 2021-2022 H. Shinaoka and others
# SPDX-License-Identifier: MIT

import numpy as np
from .matrix import DenseMatrix, DiagonalMatrix, PartialDiagonalMatrix, ScaledIdentityMatrix, asmatrixtype, matrix_hash, MatrixBase
from typing import Sequence, Union, Optional, Tuple, cast, Iterable
from scipy.sparse.linalg import lgmres, LinearOperator

add = lambda x, y: x + y
matmul = lambda x, y: x @ y
inv = lambda x: np.linalg.inv(x) if isinstance(x, np.ndarray) else x.inv()

def _assert_optional_types(obj, types):
    flag = False
    if obj is None:
        flag = True
    for t in types:
        flag = flag or isinstance(obj, t)
    assert flag

def _assert_types(obj, types):
    flag = False
    for t in types:
        flag = flag or isinstance(obj, t)
    assert flag


class ObjectiveFunctionBase(object):
    """
    Base class for objective function F(x)
    """
    def __init__(self, size_x: int) -> None:
        super().__init__()
        self._size_x = size_x

    @property
    def size_x(self) -> int:
        return self._size_x

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate F(x)"""
        return NotImplemented

    def solve(self,
        h: Union[np.ndarray,None],
        mu: Optional[MatrixBase]) -> np.ndarray:
        """
        Return argmin_x F(x) + h^+ x + x^+ h + x^+ mu x

        h: 1D array
        mu: a positive definite matrix
        """
        return NotImplemented


class LeastSquares(ObjectiveFunctionBase):
    """
    alpha * ||y - A @ x||_2^2
    """
    def __init__(
            self,
            alpha: float,
            A: Union[np.ndarray, MatrixBase],
            y: np.ndarray
        ) -> None:
        assert A.ndim == 2
        assert y.ndim == 1
        assert A.shape[0] == y.size
        _assert_types(A, [np.ndarray, MatrixBase])
        A = asmatrixtype(A)
        super().__init__(A.shape[1])

        self._alpha = alpha
        self._A = A # type: MatrixBase
        self._y = y # type: np.ndarray
        self._Ac = A.conjugate().T
        self._AcA = self._Ac @ A
        self._Nx = A.shape[1]

        # B = (A^+ A + mu)^{-1}
        self._B_cache = (0, DenseMatrix(np.zeros((1,1)))) # type: Tuple[int, MatrixBase]

    def __call__(self, x: np.ndarray) -> float:
        Ax = cast(np.ndarray, self._A @ x) # type: np.ndarray
        y = cast(np.ndarray, self._y) # type: np.ndarray
        diff = cast(np.ndarray, y - Ax) # type: np.ndarray
        return float(self._alpha * np.linalg.norm(diff)**2)

    def _get_B(self, mu: MatrixBase) -> MatrixBase:
        hash_ = matrix_hash(mu)
        if self._B_cache[0] != hash_:
            self._B_cache = (
                hash_,
                cast(MatrixBase, inv(add(self._alpha * self._AcA, mu)))
            )
        return self._B_cache[1]

    def solve(self, h: Optional[np.ndarray] = None, mu: Optional[MatrixBase] = None) -> np.ndarray:
        _assert_optional_types(h, [np.ndarray])
        _assert_optional_types(mu, [MatrixBase])

        if h is None:
            h = np.zeros(self._Nx)
        if mu is None:
            mu = DiagonalMatrix(np.zeros(self._Nx))
        assert h.shape == (self._Nx,)
        assert mu.shape == (self._Nx, self._Nx)
        vec = self._alpha * (self._Ac @ self._y) - h
        assert isinstance(vec, np.ndarray)
        return cast(np.ndarray, self._get_B(mu) @ vec)


class ConstrainedLeastSquares(LeastSquares):
    r"""
    alpha * ||y - A @ x||_2^2 + \infty * ||C@x - D||_2^2 = 0
    """
    def __init__(self,
            alpha: float,
            A: Union[np.ndarray, MatrixBase],
            y: np.ndarray,
            C: Union[np.ndarray, MatrixBase],
            D: np.ndarray
        ) -> None:
        assert A.ndim == 2
        assert y.ndim == 1
        assert C.ndim == 2
        assert D.ndim == 1
        assert A.shape[0] == y.size
        assert A.shape[1] == C.shape[1]
        assert C.shape[0] == D.size
        _assert_types(A, [np.ndarray, MatrixBase])
        _assert_types(C, [np.ndarray, MatrixBase])
        super().__init__(alpha, asmatrixtype(A), y)

        self._C = DenseMatrix(C) if isinstance(C, np.ndarray) else C # type: MatrixBase
        self._D = D

    def solve(self, h: Optional[np.ndarray] = None, mu: Optional[MatrixBase] = None) -> np.ndarray:
        _assert_optional_types(h, [np.ndarray])
        _assert_optional_types(mu, [MatrixBase])

        if h is None:
            h = np.zeros(self._Nx)
        assert h.shape == (self._Nx,)
        if mu is None:
            mu = ScaledIdentityMatrix(self._Nx, 0.0)
        assert mu.shape == (self._Nx, self._Nx)
        Ch = self._C.conjugate().T # type: MatrixBase
        B =  self._get_B(mu)
        xi1 = cast(np.ndarray, B @ (self._alpha * (self._Ac @ self._y) - h))
        assert isinstance(xi1, np.ndarray)
        xi2 = - (B @ Ch)
        tmp1 = inv(self._C @ xi2) # Union[MatrixBase, np.ndarray]
        tmp2 = cast(np.ndarray, self._D - self._C @ xi1)
        assert isinstance(tmp2, np.ndarray)
        nu = tmp1 @ tmp2
        return xi1 + xi2 @ nu


class L1Regularizer(ObjectiveFunctionBase):
    """
    L1 regularization
        F(x) = alpha * |x|_1
    """
    def __init__(self, alpha: float, size_x: int) -> None:
        assert isinstance(size_x, int), type(size_x)
        super().__init__(size_x)
        assert alpha > 0
        self._alpha = alpha

    def __call__(self, x) -> float:
        return self._alpha * np.sum(np.abs(x))

    def solve(
        self,
        h: Optional[np.ndarray] = None,
        mu: Optional[MatrixBase] = None) -> np.ndarray:
        """
        x <- argmin_x alpha * |x|_1 + h^+ x + x^+ h + mu x^+ x

        This function works only if all the following conditions are met:
          * h and x are real vectors
          * mu is a diagonal matrix
        Return a real vector.
        """
        _assert_types(h, [np.ndarray])
        assert isinstance(mu, DiagonalMatrix) or isinstance(mu, ScaledIdentityMatrix)

        if h is None:
            raise ValueError("h must not be None!")
        if mu is None:
            raise ValueError("mu must not be None!")
        if np.iscomplexobj(h):
            h = h.real
        return _softmax(-(h/mu.diagonals), 0.5*self._alpha/mu.diagonals)


class L2Regularizer(ObjectiveFunctionBase):
    """
    L2 regularization
        F(x) = alpha * |A x|_2^2
    """
    def __init__(self, alpha: float, A: Union[np.ndarray, MatrixBase]):
        _assert_optional_types(A, [np.ndarray, MatrixBase])
        A = asmatrixtype(A)
        super().__init__(A.shape[1])
        assert alpha > 0
        self._alpha = alpha
        self._A = A
        self._AcA = matmul(A.conjugate().T, A)

        # B = (A^+ A + mu)^{-1}
        self._B_cache = (0, DenseMatrix(np.zeros((1,1))))

    def __call__(self, x: np.ndarray):
        return self._alpha * np.linalg.norm(matmul(self._A, x))**2

    def _get_B(self, mu: MatrixBase):
        hash_ = matrix_hash(mu)
        if self._B_cache[0] != hash_:
            self._B_cache = (
                hash_,
                inv(add(self._alpha * self._AcA, mu))
            )
        return self._B_cache[1]

    def solve(self,
        h: Optional[np.ndarray] = None,
        mu: Optional[MatrixBase] = None):
        """
        x <- argmin_x alpha * x^+ (A^+ A) x + h^+ x + x^+ h + x^+ mu x
            = - (alpha A^+ A + mu)^{-1} h

        """
        _assert_optional_types(h, [np.ndarray])
        _assert_optional_types(mu, [MatrixBase])
        if mu is None:
            mu = ScaledIdentityMatrix(self._A.shape[1], 0.0)
        if h is None:
            return np.zeros(self._A.shape[1])
        else:
            return - (self._get_B(mu) @ h)


class NonNegativePenalty(ObjectiveFunctionBase):
    """
    Non-negative penalty term
        F(x) = infty * Theta(-x)
    """
    def __init__(self, size_x: int):
        super().__init__(size_x)

    def __call__(self, x: np.ndarray):
        return 0.0

    def solve(self, h: Optional[np.ndarray] = None, mu: Optional[MatrixBase] = None):
        """
        This function works only if all the following conditions are met:
          * h and x are real vectors
          * mu is a diagonal matrix
        Return a real vector.
        """
        assert isinstance(h, np.ndarray)
        assert isinstance(mu, DiagonalMatrix) or isinstance(mu, ScaledIdentityMatrix)
        if h is None:
            h = np.zeros(self.size_x)
        elif np.iscomplexobj(h):
            h = h.real
        if mu is None:
            raise ValueError("mu must not be None!")
        return _project_plus(-(h/mu.diagonals))


class SemiPositiveDefinitePenalty(ObjectiveFunctionBase):
    """
    Penalty term for negative eigenvalues of x

    1) Reshape x into a three-way tensor
    2) Along a given axis, we compute eigenvalues and remove negative ones (we assume hermition matrices).
    """
    def __init__(self, shape: Union[Sequence,np.ndarray], axis: int):
        assert len(shape) == 3
        super().__init__(np.prod(shape))
        self._shape = shape
        self._axis = axis

    def __call__(self, x: np.ndarray):
        return 0.0

    def solve(self, h : Optional[np.ndarray] = None, mu: Optional[MatrixBase] = None):
        """
        This function works only if mu is a diagonal matrix
        mu must be regarded as a diagonal matrix
        """
        assert isinstance(h, np.ndarray)
        assert isinstance(mu, DiagonalMatrix) or isinstance(mu, ScaledIdentityMatrix) or \
            (isinstance(mu, PartialDiagonalMatrix) and isinstance(mu.matrix, ScaledIdentityMatrix)) or \
            (isinstance(mu, PartialDiagonalMatrix) and isinstance(mu.matrix, DiagonalMatrix))
        if mu is None:
            raise ValueError("mu must not be None!")

        diagonals = None
        if isinstance(mu, DiagonalMatrix) or isinstance(mu, ScaledIdentityMatrix):
            diagonals = mu.diagonals
        elif isinstance(mu.matrix, ScaledIdentityMatrix):
            diagonals = np.full(mu.shape[0], mu.matrix.coeff)
        elif isinstance(mu.matrix, DiagonalMatrix):
            diagonals = np.einsum('i,j->ij', mu.matrix.diagonals, np.ones(np.prod(mu.rest_dims))).ravel()
        assert diagonals is not None
        assert diagonals.ndim == 1, diagonals.shape


        if h is None:
            h = np.zeros(np.prod(self._shape))
        elif np.iscomplexobj(h):
            h = h.real

        assert diagonals.size == h.size

        x_ = (-(h/diagonals)).reshape(self._shape)
        x_ = np.moveaxis(x_, self._axis, 0)
        for i in range(x_.shape[0]):
            evals, evecs = np.linalg.eigh(x_[i,:,:])
            idx = evals >= 0
            U = evecs[:, idx]
            x_[i,:,:] = U @ (evals[idx,None] * U.conjugate().T)
        return np.moveaxis(x_, 0, self._axis).ravel()


def _project_plus(x):
    res = x.copy()
    res[x < 0] = 0
    return res

def _softmax(y: np.ndarray, lambda_: np.ndarray):
    """
    Softmax function

    y - lambda_ (y >  lambda_)
    y + lambda_ (y < -lambda_)
    0           (otherwise)
    """
    assert isinstance(y, np.ndarray) and y.ndim == 1
    assert isinstance(lambda_, np.ndarray) and lambda_.ndim == 1
    assert (np.asarray(lambda_) > 0).all()
    assert y.size == lambda_.size
    res = np.zeros(y.size)

    idx = y > lambda_
    res[idx] = y[idx] - lambda_[idx]

    idx = y < -lambda_
    res[idx] = y[idx] + lambda_[idx]

    return res
