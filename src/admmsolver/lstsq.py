import numpy as np

class LeastSquaresBase(object):
    """
    Base class for main least-squares fitting problem 
        |y - A x|_2^2
    subject to the linear constraint V x = 1.
    The shares of these matrices/vectors are
        y: (Ny,)
        A: (Ny, Nx)
        V: (Nc, Nx)
    The constraint is tread by the method of Lagrange multiplier.
    We optimize
        |y - A x|_2^2 - nu^+ (V x -1) - (V x - 1)^+ nu + 
            h^+ x + x^+ h + mu x^+ x,
    where the second line denotes contributions from other penalty terms.
    """
    def __init__(self, A, y, V=None):
        assert y.ndim == 1
        assert A.ndim == 2
        self._Ny, self._Nx = A.shape
        if V is None:
            self._Nc = 0
        else:
            self._Nc = V.shape[0]
            assert V.shape == (self._Nc, self._Nx)
        assert y.shape == (self._Ny,)

        self._A = A
        self._y = y
        self._V = V
    
    def solve(self, nu, h, mu):
        """
        Return new x and nu
            x <- argmin_x |y - Ax|_2^2 + h^+ x + x^+ h + mu x^+ x
        """
        pass

    @property
    def shape(self):
        return (self._Ny, self._Nx)
    
    @property
    def num_constraints(self):
        return self._Nc


class MatrixLeastSquares(LeastSquaresBase):
    """
    Least-squares fitting problem with a dense/diagonal matrix A.
    A closed-form solution is used.
    """
    def __init__(self, A, y, V=None):
        assert type(A) in [np.ndarray, DiagonalMatrix]
        assert A.ndim == 2
        super().__init__(A, y, V)

        self._Ac = A.conjugate().T
        self._AcA = A.conjugate().T @ A

        # B = (A^+ A + mu I)^{-1}
        self._B_cache = (None, None)
    
    
    def _get_B(self, mu):
        assert mu>=0
        if self._B_cache[0] != mu:
            self._B_cache = (
                mu,
                _inv(self._AcA + mu * np.identity(self._Nx))
            )
        return self._B_cache[1]
    

    def solve(self, nu, h, mu):
        """
        Return new x and nu.
        If V is not given, an array filled with zero is returned as nu.
        """
        assert nu.shape == (self._Nc,)
        assert h.shape == (self._Nx,)

        B = self._get_B(mu)

        if self._Nc == 0:
            return B @ (self._Ac @ self._y - h), np.zeros(0)
        else:
            tildeh = h - self._V.conjugate().T @ nu
            new_x = B @ (self._Ac @ self._y - tildeh)

            xi1 = B @ (self._Ac @ self._y - h)
            xi2 = B @ self._V.T.conjugate()
            new_nu = _inv(self._V @ xi2) @ (np.identity(self._Nc) - self._V @ xi1)

            return new_x, new_nu.reshape(self._Nc)


class SparseLeastSquares(LeastSquaresBase):
    """
    Least-squares fitting problem with a sparse matrix A
    """
    pass


class DiagonalMatrix(object):
    """
    Numpy-compatible type for diagonal matrices
    """
    pass


def _inv(A):
    """ Compute inverse of a matrix A"""
    if isinstance(A, np.ndarray):
        return np.linalg.inv(A)
    elif isinstance(A, DiagonalMatrix):
        return DiagonalMatrix(1/A.diagonals)
    else:
        raise ValueError("Invalid value of A!")