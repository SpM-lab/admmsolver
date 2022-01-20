
import numpy as np
from typing import Union

class _MatrixBase(object):
    pass

    def __neg__(self):
        return -1 * self

    def __sub__(self, other):
        return self + (-other)

MatrixBase = _MatrixBase

class DenseMatrix(_MatrixBase):
    def __init__(self, matrix) -> None:
        assert matrix.ndim == 2
        self.matirx = matrix
        self.shape = matrix.shape
        self.ndim = 2
    
    def hash(self):
        return matrix_hash(self.asmatrix())
    
    def asmatrix(self):
        return self.matirx
    
    def inv(self):
        return DenseMatrix(np.linalg.inv(self.asmatrix()))
    
    @property
    def T(self):
        return DenseMatrix(self.matirx.T)

    def conjugate(self):
        return DenseMatrix(self.matirx.conjugate())
    
    conj = conjugate

    def __mul__(self, other):
        if np.isscalar(other):
            return DenseMatrix(self.matirx * other)
        else:
            return NotImplemented
    
    __rmul__ = __mul__

    def __matmul__(self, other):
        assert self.shape[1] == other.shape[0]
        assert isinstance(other, _MatrixBase) or (isinstance(other, np.ndarray) and other.ndim==1)
        if isinstance(other, np.ndarray):
            return self.asmatrix() @ other
        else:
            if isinstance(other, ScaledIdentityMatrix):
                return DenseMatrix(self.matirx * other.coeff)
            elif isinstance(other, DiagonalMatrix):
                return DenseMatrix(self.matirx * other.diagonals[None,:])
            else:
                return DenseMatrix(self.matirx @ other.asmatrix())

    def __add__(self, other):
        assert isinstance(other, _MatrixBase)
        assert self.shape == other.shape

        return DenseMatrix(self.asmatrix() + other.asmatrix())



class ScaledIdentityMatrix(_MatrixBase):
    """
    Scaled Identity matrix
    """
    def __init__(self, size: int, coeff)->None:
        assert np.issubdtype(type(size), np.integer)
        assert np.isscalar(coeff)
        self.size = size
        self.coeff = coeff
        self.ndim = 2
        self.shape = (size, size)

    def hash(self):
        return matrix_hash(self.coeff)
    
    def asmatrix(self):
        return self.coeff * np.identity(self.size)

    def inv(self):
        return ScaledIdentityMatrix(self.size, 1/self.coeff)

    @property
    def T(self):
        return self

    @property
    def diagonals(self):
        return np.full(self.size, self.coeff)

    def conjugate(self):
        return ScaledIdentityMatrix(self.size, np.conjugate(self.coeff))
    
    conj = conjugate

    def __mul__(self, other):
        if np.isscalar(other):
            return ScaledIdentityMatrix(self.size, self.coeff*other)
        else:
            return NotImplemented
    
    __rmul__ = __mul__
 
    def __matmul__(self, other: Union[_MatrixBase, np.ndarray]):
        assert self.shape[1] == other.shape[0], f"{self.shape} {other.shape}"
        assert isinstance(other, _MatrixBase) or (isinstance(other, np.ndarray) and other.ndim==1)
        return self.coeff * other

    def __add__(self, other):
        assert isinstance(other, _MatrixBase)
        assert self.shape == other.shape, f"{self.shape} {other.shape}"

        if isinstance(other, DenseMatrix):
            return DenseMatrix(self.asmatrix() + other.asmatrix())
        elif isinstance(other, ScaledIdentityMatrix):
            return ScaledIdentityMatrix(self.size, self.coeff + other.coeff)
        elif isinstance(other, DiagonalMatrix):
            return DiagonalMatrix(self.coeff*np.ones(self.size) + other.diagonals)
        elif isinstance(other, PartialDiagonalMatrix):
            return PartialDiagonalMatrix(
                ScaledIdentityMatrix(other.matrix.shape[0], self.coeff) + other.matrix,
                other.rest_dims)
        else:
            return NotImplemented


class DiagonalMatrix(_MatrixBase):
    """
    Diagonal matrix
    """
    def __init__(self, diagonals):
        assert diagonals.ndim == 1
        self._diagonals = diagonals
        self.ndim = 2
        self.shape = (diagonals.size, diagonals.size)

    def hash(self):
        return matrix_hash(self.diagonals)
    
    @property
    def diagonals(self):
        return self._diagonals
    
    def inv(self):
        return DiagonalMatrix(1/self.diagonals)

    def asmatrix(self):
        return np.diag(self._diagonals)

    @property
    def T(self):
        return self
    
    def conjugate(self):
        return DiagonalMatrix(self.diagonals.conjugate())
    
    conj = conjugate

    def __mul__(self, other):
        if np.isscalar(other):
            return DiagonalMatrix(self._diagonals * other)
        else:
            return NotImplemented
        
    __rmul__ = __mul__

    def __add__(self, other):
        assert isinstance(other, _MatrixBase)
        assert self.shape == other.shape

        if isinstance(other, DenseMatrix):
            return DenseMatrix(other.asmatrix() + np.diag(self.diagonals))
        elif isinstance(other, ScaledIdentityMatrix):
            return DiagonalMatrix(self.diagonals + np.full(self.diagonals.size, other.coeff))
        elif isinstance(other, DiagonalMatrix):
            return DiagonalMatrix(self.diagonals + other.diagonals)
        elif isinstance(other, PartialDiagonalMatrix):
            return DenseMatrix(self.asmatrix() + other.asmatrix())
        else:
            return NotImplemented


    def __matmul__(self, other: Union[_MatrixBase, np.ndarray]):
        """ self @ other """
        assert self.shape[1] == other.shape[0]
        assert isinstance(other, _MatrixBase) or (isinstance(other, np.ndarray) and other.ndim==1)

        if isinstance(other, np.ndarray):
            return self._diagonals * other
        elif isinstance(other, DenseMatrix):
            return DenseMatrix(self._diagonals[:,None] * other.matirx)
        elif isinstance(other, DiagonalMatrix):
            return DiagonalMatrix(self._diagonals * other._diagonals)
        elif isinstance(other, PartialDiagonalMatrix):
            return DenseMatrix(self._diagonals[:,None] * other.asmatrix())
        else:
            return NotImplemented

    def __str__(self):
        return "DiagonalMatrix: " + self.diagonals.__str__()


class PartialDiagonalMatrix(_MatrixBase):
    """
    Matrix that can be composed as
        A otimes I.
    """
    def __init__(self, matrix: Union[np.ndarray, _MatrixBase], rest_dims: tuple):
        assert matrix.ndim == 2
        self.matrix = asmatrixtype(matrix)
        self._matrix_cg = matrix.T.conj()
        self.rest_dims = rest_dims
        self.ndim = 2
        self.shape = (matrix.shape[0]*np.prod(rest_dims), matrix.shape[1]*np.prod(rest_dims))

    def hash(self):
        return matrix_hash(self.matrix)
    
    def asmatrix(self):
        return np.einsum(
            'IJ,ij->IiJj',
            self.matrix.asmatrix(),
            np.identity(np.prod(self.rest_dims)),
            optimize=True
        ).reshape(self.shape)
    
    def inv(self):
        return PartialDiagonalMatrix(self.matrix.inv(), self.rest_dims)

    @property
    def T(self):
        return PartialDiagonalMatrix(self.matrix.T, self.rest_dims)

    def conjugate(self):
        return PartialDiagonalMatrix(self.matrix.conjugate(), self.rest_dims)

    conj = conjugate

    def __matmul__(self, other):
        """ self @ other """
        assert self.shape[1] == other.shape[0]
        assert isinstance(other, _MatrixBase) or (isinstance(other, np.ndarray) and other.ndim==1)
        if isinstance(other, np.ndarray):
            return self.matvec(other)
        elif isinstance(other, PartialDiagonalMatrix) and self.rest_dims == other.rest_dims:
            return PartialDiagonalMatrix(self.matrix@other.matrix, self.rest_dims)
        else:
            return DenseMatrix(self.asmatrix() @ other.asmatrix())


    def __mul__(self, other):
        if np.isscalar(other):
            return PartialDiagonalMatrix(self.matrix * other, self.rest_dims)
        else:
            return NotImplemented

    __rmul__ = __mul__
    
    def __add__(self, other):
        assert isinstance(other, _MatrixBase)
        assert self.shape == other.shape

        if isinstance(other, ScaledIdentityMatrix):
            return PartialDiagonalMatrix(
                self.matrix + ScaledIdentityMatrix(self.matrix.shape[0], other.coeff), self.rest_dims)
        else:
            return DenseMatrix(self.asmatrix() + other.asmatrix())

    
    def matvec(self, v):
        """
        (a \otimes I) @ v
        v can be a vector or a tensor.
        In the latter case, the matrix applied to the first axis of v.
        """
        v = v.reshape(self.matrix.shape[1], *self.rest_dims, -1)
        if isinstance(self.matrix, DiagonalMatrix):
            return np.einsum('d,d...->d...', self.matrix.diagonals, v).ravel()
        elif isinstance(self.matrix, DenseMatrix):
            return np.tensordot(
                self.matrix.asmatrix(),
                v,
                axes=(-1,0)
            ).ravel()
        else:
            raise RuntimeError(f"Unsupported type{type(self.matrix)}!")

    def rmatvec(self, v):
        """ (a \otimes I)^dagger @ v"""
        return np.tensordot(
            self._matrix_cg,
            v.reshape(-1, *self.rest_dims),
            axes=(-1,0)
        ).ravel()
    

def identity(n, dtype=np.float64):
    """ Create an identity matrix """
    return ScaledIdentityMatrix(n, dtype(1.0))


def matrix_hash(a):
    """ Compute hash of a matrix a"""
    if isinstance(a, np.ndarray):
        return hash(a.data.tobytes()) # This makes a copy
    elif np.isscalar(a):
        return hash(a)
    else:
        return a.hash()

#def inv(a):
    #""" Compute inverse of a matrix a"""
    #assert isinstance(a, _MatrixBase)
    #if isinstance(a, DenseMatrix):
        #r = DenseMatrix(np.linalg.inv(a.matrix))
    #elif isinstance(a, ScaledIdentityMatrix):
        #r = ScaledIdentityMatrix(a.size, 1/a.coeff)
    #elif isinstance(a, DiagonalMatrix):
        #r = DiagonalMatrix(1/a.diagonals)
    #elif isinstance(a, PartialDiagonalMatrix):
        #r = PartialDiagonalMatrix(np.linalg.inv(a.matrix), a.rest_dims)
    #return r


#def diagonal(a):
    #""" Return diagonal elements as a 1D array """
    #if isinstance(a, np.ndarray) and a.ndim == 2:
        #return np.diag(a)
    #elif isinstance(a, DiagonalMatrix):
        #return a.diagonals
    #else:
        #raise ValueError("Invalid value of A!")

#def matmul(a, b):
    #"""Matrix multiplication"""
    #if a.ndim == 2:
        #assert a.shape[1] == b.shape[0], f"{a.shape}, {b.shape}"
    
    #if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        #r = a @ b
    #elif isinstance(a, ScaledIdentityMatrix) or isinstance(b, ScaledIdentityMatrix):
        #r = a.__matmul__(b)
    #elif isinstance(a, DiagonalMatrix):
        #r = a.__matmul__(b)
    #elif isinstance(a, PartialDiagonalMatrix):
        #r = a.__matmul__(b)
    #elif isinstance(a, np.ndarray) and isinstance(b, DiagonalMatrix):
        #if a.ndim == 1:
            #r = a * b.diagonals
        #elif a.ndim == 2:
            #r = a * b.diagonals[None,:]
    #else:
        #raise ValueError("No way to perform matmul!")
    #assert type(r) != type(NotImplemented)
    #return r
    #

def asmatrixtype(a):
    assert isinstance(a, _MatrixBase) or (isinstance(a, np.ndarray) and a.ndim==2)
    if isinstance(a, np.ndarray):
        return DenseMatrix(a)
    return a

#def asmatrix(a):
    #if isinstance(a, np.ndarray):
        #return a
    #else:
        #return a.asmatrix()

#def add(a, b):
#    """Add two matrix objects"""
#    r = None
#    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
#        r = a + b
#    elif isinstance(a, DiagonalMatrix):
#        r = a.__add__(b)
#    elif isinstance(a, PartialDiagonalMatrix):
#        r = a.__add__(b)
#    elif isinstance(b, DiagonalMatrix):
#        r = b.__add__(a)
#
#    if type(r) == type(NotImplemented) or r is None:
#        r = asmatrix(a) + asmatrix(b)
#    
#    return r