import numpy as np
from .matrix import DiagonalMatrix, inv, matrix_hash, diagonal, add, matmul

class ObjectiveFunctionBase(object):
    """
    Base class for objective function F(x)
    """
    def __init__(self, size_x):
        super().__init__()
        self._size_x = size_x
    
    @property
    def size_x(self):
        return self._size_x

    def __call__(self, x):
        """Evaluate F(x)"""
        raise NotImplementedError()
    
    def solve(self, h, mu):
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
    def __init__(self, alpha, A, y):
        assert A.ndim == 2
        super().__init__(A.shape[1])

        self._alpha = alpha
        self._A = A
        self._y = y
        self._Ac = A.conjugate().T
        self._AcA = A.conjugate().T @ A
        self._Nx = A.shape[1]

        # B = (A^+ A + mu)^{-1}
        self._B_cache = (None, None)
    
    def __call__(self, x):
        return self._alpha * np.linalg.norm(self._y - self._A @ x)**2
    
    def _get_B(self, mu):
        hash_ = matrix_hash(mu)
        if self._B_cache[0] != hash_:
            #x = self._alpha * self._AcA
            self._B_cache = (
                hash_,
                inv(add(self._alpha * self._AcA, mu))
            )
        return self._B_cache[1]
    
    def solve(self, h, mu):
        if h is None:
            h = np.zeros(self._Nx)
        if mu is None:
            mu = DiagonalMatrix(np.zeros(self._Nx))
        assert h.shape == (self._Nx,)
        assert mu.shape == (self._Nx, self._Nx)
        return self._get_B(mu) @ (self._alpha * self._Ac @ self._y - h)


class ConstrainedLeastSquares(LeastSquares):
    """
    alpha * ||y - A @ x||_2^2 + \infty * ||C@x - D||_2^2 = 0
    """
    def __init__(self, alpha, A, y, C, D):
        assert A.ndim == 2
        super().__init__(alpha, A, y)

        self._C = C
        self._D = D

    def solve(self, h, mu):
        assert h.shape == (self._Nx,)
        assert mu.shape == (self._Nx, self._Nx)
        B =  self._get_B(mu)
        xi1 = B @ (self._alpha * self._Ac @ self._y - h)
        xi2 = - B @ self._C.conjugate().T
        nu = inv(self._C @ xi2) @ (self._D - self._C @ xi1)
        return xi1 + xi2 @ nu


class L1Regularizer(ObjectiveFunctionBase):
    """
    L1 regularization
        F(x) = alpha * |x|_1
    """
    def __init__(self, alpha, size_x):
        super().__init__(size_x)
        assert alpha > 0
        self._alpha = alpha
    
    def __call__(self, x):
        return self._alpha * np.sum(np.abs(x))

    def solve(self, h, mu):
        """
        x <- argmin_x alpha * |x|_1 + h^+ x + x^+ h + mu x^+ x

        This function works only if all the following conditions are met:
          * h and x are real vectors
          * mu is a diagonal matrix
        Return a real vector.
        """
        assert isinstance(mu, DiagonalMatrix)
        if np.iscomplexobj(h):
            h = h.real
        return _softmax(-h/mu.diagonals, 0.5*self._alpha/mu.diagonals)


class L2Regularizer(ObjectiveFunctionBase):
    """
    L2 regularization
        F(x) = alpha * |A x|_2^2
    """
    def __init__(self, alpha, A):
        super().__init__(A.shape[1])
        assert alpha > 0
        self._alpha = alpha
        self._A = A
        self._AcA = matmul(A.conjugate().T, A)

        # B = (A^+ A + mu)^{-1}
        self._B_cache = (None, None)
    
    def __call__(self, x):
        return self._alpha * np.linalg.norm(matmul(self._A, x))**2
    
    def _get_B(self, mu):
        hash_ = matrix_hash(mu)
        if self._B_cache[0] != hash_:
            self._B_cache = (
                hash_,
                inv(add(self._alpha * self._AcA, mu))
            )
        return self._B_cache[1]

    def solve(self, h, mu):
        """
        x <- argmin_x alpha * x^+ (A^+ A) x + h^+ x + x^+ h + x^+ mu x
          = - (alpha A^+ A + mu)^{-1} h

        """
        return matmul(self._get_B(mu), -h)



class NonNegativePenalty(ObjectiveFunctionBase):
    """
    Non-negative penalty term
        F(x) = infty * Theta(-x)
    """
    def __init__(self, size_x):
        super().__init__(size_x)
    
    def __call__(self, x):
        return 0.0

    def solve(self, h, mu):
        """
        This function works only if all the following conditions are met:
          * h and x are real vectors
          * mu is a diagonal matrix
        Return a real vector.
        """
        assert isinstance(mu, DiagonalMatrix)
        if np.iscomplexobj(h):
            h = h.real
        return _project_plus(-h/mu.diagonals)


class SemiPositiveDefinitePenalty(ObjectiveFunctionBase):
    """
    Penalty term for negative eigenvalues of x

    1) Reshape x into a three-way tensor
    2) Along a given axis, we compute eigenvalues and remove negative ones (we assume hermition matrices).
    """
    def __init__(self, shape_x, axis):
        assert len(shape_x) == 3
        super().__init__(np.prod(shape_x))
        self._shape_x = shape_x
        self._axis = axis
    
    def __call__(self, x):
        return 0.0

    def solve(self, h, mu):
        """
        This function works only if mu is a diagonal matrix
        """
        assert isinstance(mu, DiagonalMatrix)
        if np.iscomplexobj(h):
            h = h.real
        x_ = (-h/mu.diagonals).reshape(self._shape_x)
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

def _softmax(y, lambda_):
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