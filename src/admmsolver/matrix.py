
import numpy as np

class DiagonalMatrix(object):
    """
    Diagonal matrix
    """
    def __init__(self, diagonals):
        assert diagonals.ndim == 1
        self._diagonals = diagonals
        self.ndim = 2
        self.shape = (diagonals.size, diagonals.size)
    
    @property
    def diagonals(self):
        return self._diagonals

    @property
    def T(self):
        return self
    
    def conjugate(self):
        return DiagonalMatrix(self.diagonals.conjugate())

    def __neg__(self):
        return -1 * self

    def __mul__(self, other):
        if np.isscalar(other):
            return DiagonalMatrix(self._diagonals * other)
        else:
            return NotImplemented
        
    __rmul__ = __mul__

    def __add__(self, other):
        if isinstance(other, np.ndarray):
            assert self.shape == other.shape 
            return other + np.diag(self.diagonals)
        elif isinstance(other, DiagonalMatrix):
            return DiagonalMatrix(self.diagonals + other.diagonals)
        else:
            return NotImplemented

    def __sub__(self, other):
        return self + (-other)

    def __iadd__(self, other):
        if isinstance(other, DiagonalMatrix):
            self._diagonals += other.diagonals
            return self
        else:
            return NotImplemented
    
    def __matmul__(self, other):
        """ self @ other """
        if isinstance(other, np.ndarray):
            if other.ndim == 1:
                return self._diagonals * other
            elif other.ndim == 2:
                return self._diagonals[:,None] * other
            else:
                raise ValueError("ndim > 2 is not supported!")
        elif isinstance(other, DiagonalMatrix):
            return DiagonalMatrix(self._diagonals * other._diagonals)
        else:
            return NotImplemented

    def __rmatmul__(self, other):
        """ other @ self """
        if isinstance(other, np.ndarray):
            if other.ndim == 1:
                return DiagonalMatrix(self._diagonals * other)
            elif other.ndim == 2:
                return other * self._diagonals[None,:]
            else:
                raise ValueError("ndim > 2 is not supported!")
        else:
            return NotImplemented
    
    def __str__(self):
        return "DiagonalMatrix: " + self.diagonals.__str__()


class PartialDiagonalMatrix(object):
    """
    Matrix that can be composed as
        A otimes I.
    """
    def __init__(self, matrix: np.ndarray, rest_dims: tuple):
        assert matrix.ndim == 2
        self.matrix = matrix
        self._matrix_cg = matrix.T.conj()
        self.rest_dims = rest_dims
        self.ndim = 2
        self.shape = (matrix.shape[0]*np.prod(rest_dims), matrix.shape[1]*np.prod(rest_dims))

    @property
    def T(self):
        return PartialDiagonalMatrix(self.matrix.T, self.rest_dims)

    def conjugate(self):
        return PartialDiagonalMatrix(self.matrix.conjugate(), self.rest_dims)

    def __matmul__(self, other):
        """ self @ other """
        if isinstance(other, np.ndarray):
            return self.matvec(other)
        elif isinstance(other, PartialDiagonalMatrix):
            assert self.rest_dims == other.rest_dims
            return PartialDiagonalMatrix(self.matrix @ other.matrix, self.rest_dims)
        else:
            return NotImplemented
    
    def matvec(self, v):
        """
        (a \otimes I) @ v
        v can be a vector or a tensor.
        In the latter case, the matrix applied to the first axis of v.
        """
        return np.tensordot(
            self.matrix,
            v.reshape(self.matrix.shape[1], *self.rest_dims, -1),
            axes=(-1,0)
        ).ravel()

    def rmatvec(self, v):
        """ (a \otimes I)^dagger @ v"""
        return np.tensordot(
            self._matrix_cg,
            v.reshape(-1, *self.rest_dims),
            axes=(-1,0)
        ).ravel()
    

def identity(n, dtype=np.float64):
    """ Create a DiagonalMatrix instance representing an identity matrix of size n """
    return DiagonalMatrix(np.ones(n, dtype=np.float64))


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
        raise ValueError(f"Invalid type{type(a)} of a!")

def diagonal(a):
    """ Return diagonal elements as a 1D array """
    if isinstance(a, np.ndarray) and a.ndim == 2:
        return np.diag(a)
    elif isinstance(a, DiagonalMatrix):
        return a.diagonals
    else:
        raise ValueError("Invalid value of A!")

def matmul(a, b):
    """Matrix multiplication"""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a @ b
    elif isinstance(a, DiagonalMatrix):
        return a.__matmul__(b)
    elif isinstance(a, np.ndarray) and isinstance(b, DiagonalMatrix):
        if a.ndim == 1:
            return a * b.diagonals
        elif a.ndim == 2:
            return a * b.diagonals[None,:]
    else:
        raise ValueError("No way to perform matmul!")
    

def add(a, b):
    """Add two matrix objects"""
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        return a + b
    elif isinstance(a, DiagonalMatrix):
        return a.__add__(b)
    elif isinstance(b, DiagonalMatrix):
        return b.__add__(a)
    else:
        raise ValueError("No way to add two matrices!")