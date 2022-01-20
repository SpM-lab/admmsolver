from operator import matmul
from ssl import SSLSyscallError
import numpy as np
from .matrix import DenseMatrix, DiagonalMatrix, PartialDiagonalMatrix, ScaledIdentityMatrix, asmatrixtype, matrix_hash, MatrixBase
from typing import Union, Optional
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
        raise NotImplementedError()
    
    def solve(self,
        h: Union[np.ndarray,None],
        mu: MatrixBase) -> np.ndarray:
        """
        Return argmin_x F(x) + h^+ x + x^+ h + x^+ mu x

        h: 1D array
        mu: a positive definite matrix
        """
        raise NotImplementedError()


class LeastSquares(ObjectiveFunctionBase):
    """
    alpha * ||y - A @ x||_2^2
    """
    def __init__(
            self,
            alpha: float,
            A: Union[np.ndarray, MatrixBase],
            y: np.ndarray,
            isolver: bool = False
        ) -> None:
        assert A.ndim == 2
        assert y.ndim == 1
        assert A.shape[0] == y.size
        _assert_types(A, [np.ndarray, MatrixBase])
        A = asmatrixtype(A)
        super().__init__(A.shape[1])

        self._alpha = alpha
        self._A = A
        self._y = y
        self._Ac = A.conjugate().T
        self._AcA = self._Ac @ A
        self._Nx = A.shape[1]
        self._isolver = isolver

        # B = (A^+ A + mu)^{-1}
        self._B_cache = (None, None)
    
    def __call__(self, x: np.ndarray) -> float:
        return self._alpha * np.linalg.norm(self._y - self._A @ x)**2
    
    def _get_B(self, mu: Union[np.ndarray, DiagonalMatrix]) -> Union[np.ndarray, DiagonalMatrix]:
        hash_ = matrix_hash(mu)
        if self._B_cache[0] != hash_:
            self._B_cache = (
                hash_,
                inv(add(self._alpha * self._AcA, mu))
            )
        return self._B_cache[1]
    
    def solve(self, h: Optional[np.ndarray], mu: Optional[MatrixBase]) -> np.ndarray:
        _assert_optional_types(h, [np.ndarray])
        _assert_optional_types(mu, [MatrixBase])

        if h is None:
            h = np.zeros(self._Nx)
        if mu is None:
            mu = DiagonalMatrix(np.zeros(self._Nx))
        assert h.shape == (self._Nx,)
        assert mu.shape == (self._Nx, self._Nx)
        vec = self._alpha * self._Ac @ self._y - h
        if self._isolver:
            return _isolve(self._alpha, self._AcA, mu, vec)
        else:
            return self._get_B(mu) @ vec


class ConstrainedLeastSquares(LeastSquares):
    """
    alpha * ||y - A @ x||_2^2 + \infty * ||C@x - D||_2^2 = 0
    """
    def __init__(self,
        alpha: float,
        A: Union[np.ndarray, MatrixBase],
        y: np.ndarray,
        C: Union[np.ndarray, MatrixBase],
        D: np.ndarray,
        isolver: bool = False
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
        A = asmatrixtype(A)
        C = asmatrixtype(C)
        super().__init__(alpha, A, y)

        self._C = C
        self._D = D
        self._isolver = isolver

    def solve(self, h: Optional[np.ndarray], mu: Optional[np.ndarray]) -> np.ndarray:
        _assert_optional_types(h, [np.ndarray])
        _assert_optional_types(mu, [MatrixBase])

        assert h.shape == (self._Nx,)
        assert mu.shape == (self._Nx, self._Nx)
        if self._isolver:
            xi1 = _isolve(self._alpha, self._AcA, mu, self._alpha * (self._Ac @ self._y) - h)
            xi2 = _isolve(self._alpha, self._AcA, mu, -self._C.conjugate().T)
        else:
            B =  self._get_B(mu)
            xi1 = B @ (self._alpha * self._Ac @ self._y - h)
            xi2 = - B @ self._C.conjugate().T
        nu = inv(self._C @ xi2) @ (self._D - self._C @ xi1)
        return xi1 + xi2 @ nu
    
def _isolve(alpha, mat, mu, b: Union[np.ndarray, DenseMatrix]):
    """
    Solve (alpha * mat + mu * I)^{-1} @ b,
    wehre v is a matrix.
    """
    _assert_types(b, [np.ndarray, DenseMatrix])
    if isinstance(b, DenseMatrix):
        b = b.asmatrix()
    b_ = b
    if b_.ndim == 1:
        b_ = b_[:,None]

    def matvec(v):
        v = v.reshape(b_.shape)
        return alpha * (mat @ v).ravel() + (mu @ v).ravel()

    op = LinearOperator(
            dtype=np.complex128,
            shape=tuple(np.array(mat.shape) * b_.shape[1]),
            matvec=matvec
        )
    res = lgmres(op, b.ravel())[0]
    if b.ndim == 1:
        res = res.ravel()
    else:
        res = res.reshape(-1, b.shape[1])
    return res


class L1Regularizer(ObjectiveFunctionBase):
    """
    L1 regularization
        F(x) = alpha * |x|_1
    """
    def __init__(self, alpha, size_x) -> None:
        super().__init__(size_x)
        assert alpha > 0
        self._alpha = alpha
    
    def __call__(self, x) -> float:
        return self._alpha * np.sum(np.abs(x))

    def solve(self, h: Optional[np.ndarray], mu: Union[None, DiagonalMatrix, ScaledIdentityMatrix]) -> np.ndarray:
        """
        x <- argmin_x alpha * |x|_1 + h^+ x + x^+ h + mu x^+ x

        This function works only if all the following conditions are met:
          * h and x are real vectors
          * mu is a diagonal matrix
        Return a real vector.
        """
        _assert_optional_types(h, [np.ndarray])
        _assert_types(mu, [type(None), DiagonalMatrix, ScaledIdentityMatrix])
        if np.iscomplexobj(h):
            h = h.real
        return _softmax(-h/mu.diagonals, 0.5*self._alpha/mu.diagonals)


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
        self._B_cache = (None, None)
    
    def __call__(self, x: np.ndarray):
        return self._alpha * np.linalg.norm(matmul(self._A, x))**2
    
    def _get_B(self, mu: Union[np.ndarray, DiagonalMatrix]):
        hash_ = matrix_hash(mu)
        if self._B_cache[0] != hash_:
            self._B_cache = (
                hash_,
                inv(add(self._alpha * self._AcA, mu))
            )
        return self._B_cache[1]

    def solve(self,
        h: Optional[np.ndarray],
        mu: Union[None, MatrixBase]):
        """
        x <- argmin_x alpha * x^+ (A^+ A) x + h^+ x + x^+ h + x^+ mu x
          = - (alpha A^+ A + mu)^{-1} h

        """
        _assert_optional_types(h, [np.ndarray])
        _assert_optional_types(mu, [MatrixBase])
        return matmul(self._get_B(mu), -h)


class NonNegativePenalty(ObjectiveFunctionBase):
    """
    Non-negative penalty term
        F(x) = infty * Theta(-x)
    """
    def __init__(self, size_x: int):
        super().__init__(size_x)
    
    def __call__(self, x: np.ndarray):
        return 0.0

    def solve(self, h: np.ndarray, mu: Union[DiagonalMatrix, ScaledIdentityMatrix]):
        """
        This function works only if all the following conditions are met:
          * h and x are real vectors
          * mu is a diagonal matrix
        Return a real vector.
        """
        _assert_types(h, [np.ndarray])
        _assert_types(mu, [DiagonalMatrix, ScaledIdentityMatrix])
        if np.iscomplexobj(h):
            h = h.real
        return _project_plus(-h/mu.diagonals)


class SemiPositiveDefinitePenalty(ObjectiveFunctionBase):
    """
    Penalty term for negative eigenvalues of x

    1) Reshape x into a three-way tensor
    2) Along a given axis, we compute eigenvalues and remove negative ones (we assume hermition matrices).
    """
    def __init__(self, shape: np.ndarray, axis: int):
        assert len(shape) == 3
        super().__init__(np.prod(shape))
        self._shape = shape
        self._axis = axis
    
    def __call__(self, x: np.ndarray):
        return 0.0

    def solve(self, h : np.ndarray, mu: Union[DiagonalMatrix, ScaledIdentityMatrix]):
        """
        This function works only if mu is a diagonal matrix
        mu must be regarded as a diagonal matrix
        """
        assert isinstance(h, np.ndarray)
        _assert_types(mu, [DiagonalMatrix, ScaledIdentityMatrix])

        if np.iscomplexobj(h):
            h = h.real
        x_ = (-h/mu.diagonals).reshape(self._shape)
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