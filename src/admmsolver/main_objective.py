import numpy as np
from numpy.lib.arraysetops import isin


class MainLestSquaresBase(object):
    """
    Base class for main least-squares fitting problem with linear constraint
        |y - A x|_2^2
    subject to the constraint V x = 1.
    The shares of these matrices/vectors are
        y: (Ny,)
        A: (Ny, Nx)
        V: (Nc, Nx)
    The constraint is tread by the method of Lagrange multiplier.
    We optimize
        |y - A x|_2^2 - nu^+ (V x -1) - (V x - 1)^+ nu
    """
    def __init__(self, A, y, V):
        assert y.ndim == 1
        assert y.ndim == 2
        self._Ny, self._Nx = A.shape
        self._Nc = V.shape[0]
        assert y.shape == (self._Ny,)
        assert V.shape == (self._Nc, self._Nx)

        self._A = A
        self._y = y
        self._V = V
    
    def solve_for_x(self, nu, h, mu):
        """
        Return new x
         = argmin_x |y - Ax|_2^2 + h^+ x + x^+ h + mu x^+ x
        """
        pass
    
    def solve_for_nu(self, y, nu, h, mu):
        """
        Return new nu

        xi1 = (A^+ A + mu I)^{-1} (A^+ y - h + V^+ nu)
        xi2 = (A^+ A + mu I)^{-1} V
        nu <- (V xi2)^{-1} (1 - V xi1)
        """
        pass


class MainLestSquares(MainLestSquaresBase):
    """
    Main least-squares fitting problem with a dense/diagonal matrix A.
    """
    def __init__(self, A, y, V):
        assert type(A) in [np.ndarray, DiagonalMatrix]
        assert A.ndim == 2
        super().__(A, y, V)

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
    
    def solve_for_x(self, nu, h, mu):
        """
        Return new x
        """
        return self._get_B(mu) @ (self._Ac @ self._y - self._h)

    def solve_for_nu(self, y, nu, h, mu):
        """
        Return new nu
        """
        B = self._get_B(mu)
        xi1 = B @ (self._Ac @ y - h + self._V.conjugate().T @ nu)
        xi2 = B @ self._V
        return np.inv(self._V @ xi2) @ (np.identity(self._Nc) - self._V @ xi1)


class SparseMainLestSquares(MainLestSquaresBase):
    """
    Main least-squares fitting problem with a sparse matrix A
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
        return np.inv(A)
    elif isinstance(A, DiagonalMatrix):
        return DiagonalMatrix(1/A.diagonals)
    else:
        raise ValueError("Invalid value of A!")