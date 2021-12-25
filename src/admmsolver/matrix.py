
import numpy as np

class DiagonalMatrix(object):
    """
    Diagonal matrix
    """
    def __init__(self, diagonals):
        assert diagonals.ndim == 1
        self._diagonals = diagonals
        self.shape = (diagonals.size, diagonals.size)
    
    @property
    def diagonals(self):
        return self._diagonals
    
    def __mul__(self, other):
        if np.isscalar(other):
            return DiagonalMatrix(self._diagonals * other)
        else:
            return NotImplemented
    
    def __matmul__(self, other):
        if isinstance(other, np.ndarray):
            if other.ndim == 1:
                return DiagonalMatrix(self._diagonals * other)
            elif other.ndim == 2:
                return self._diagonals[:,None] * other
            else:
                raise ValueError("ndim > 2 is not supported!")
        elif isinstance(other, DiagonalMatrix):
            return DiagonalMatrix(self._diagonals * other._diagonals)
        else:
            return NotImplemented

def matrix_hash(a):
    """ Compute hash of a matrix a"""
    if isinstance(a, np.ndarray):
        return hash(a.data.tobytes()) # This makes a copy
    elif isinstance(a, DiagonalMatrix):
        return matrix_hash(a.diagonals)
    else:
        return ValueError("Not supported!")

def inv(a):
    """ Compute inverse of a matrix a"""
    if isinstance(a, np.ndarray):
        return np.linalg.inv(a)
    elif isinstance(a, DiagonalMatrix):
        return DiagonalMatrix(1/a.diagonals)
    else:
        raise ValueError("Invalid value of A!")