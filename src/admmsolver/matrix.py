
import numpy as np
from typing import Union

class MatrixBase(object):
    def __init__(self) -> None:
        super().__init__()

        self.shape = (1,1)
        self.ndim = 2

    def asmatrix(self)->np.ndarray:
        pass

    def __mul__(self, other: Union[float, complex, np.float64, np.complex128])->'MatrixBase':
        pass

    __rmul__ = __mul__

    def __matmul__(self, other: Union['MatrixBase', np.ndarray])->Union['MatrixBase',np.ndarray]:
        pass
    
    def conjugate(self)->'MatrixBase':
        pass

    @property
    def T(self)->'MatrixBase':
        pass

    @property
    def diagonals(self)->np.ndarray:
        pass

    conj = conjugate

    def __neg__(self)->'MatrixBase':
        return -1.0 * self

    def __sub__(self, other)->'MatrixBase':
        return self + (-other)

    def inv(self)->'MatrixBase':
        pass

    def hash(self)->int:
        pass

class DenseMatrix(MatrixBase):
    def __init__(self, matrix) -> None:
        assert matrix.ndim == 2
        self.matirx = matrix
        self.shape = matrix.shape
        self.ndim = 2
    
    def hash(self)->int:
        return matrix_hash(self.asmatrix())
    
    def asmatrix(self)->np.ndarray:
        return self.matirx
    
    def inv(self)->'DenseMatrix':
        return DenseMatrix(np.linalg.inv(self.asmatrix()))
    
    @property
    def T(self):
        return DenseMatrix(self.matirx.T)

    def conjugate(self):
        return DenseMatrix(self.matirx.conjugate())
    
    conj = conjugate

    def __mul__(self, other)->'DenseMatrix':
        if np.isscalar(other):
            return DenseMatrix(self.matirx * other)
        else:
            return NotImplemented
    
    __rmul__ = __mul__

    def __matmul__(self, other)->Union['DenseMatrix', np.ndarray]:
        assert self.shape[1] == other.shape[0]
        assert isinstance(other, MatrixBase) or (isinstance(other, np.ndarray) and other.ndim<=2)
        if isinstance(other, np.ndarray):
            return self.asmatrix() @ other
        else:
            if isinstance(other, ScaledIdentityMatrix):
                return DenseMatrix(self.matirx * other.coeff)
            elif isinstance(other, DiagonalMatrix):
                return DenseMatrix(self.matirx * other.diagonals[None,:])
            else:
                return DenseMatrix(self.matirx @ other.asmatrix())

    def __add__(self, other)->'DenseMatrix':
        assert isinstance(other, MatrixBase)
        assert self.shape == other.shape

        return DenseMatrix(self.asmatrix() + other.asmatrix())



class ScaledIdentityMatrix(MatrixBase):
    """
    Scaled Identity matrix
    """
    def __init__(
            self, size: int,
            coeff: Union[complex, float, np.float64, np.complex128]
        )->None:
        assert np.issubdtype(type(size), np.integer)
        assert type(coeff) in [complex, float, np.float64, np.complex128], type(coeff)
        self.size = size # type: int
        self.coeff = coeff # type: Union[complex, float, np.float64, np.complex128]
        self.ndim = 2 # type: int
        self.shape = (size, size)

    def hash(self)->int:
        return matrix_hash(self.coeff)
    
    def asmatrix(self) -> np.ndarray:
        return self.coeff * np.identity(self.size)

    def inv(self)->'ScaledIdentityMatrix':
        return ScaledIdentityMatrix(self.size, 1/self.coeff)

    @property
    def T(self)->'ScaledIdentityMatrix':
        return self

    @property
    def diagonals(self)->np.ndarray:
        return np.full(self.size, self.coeff)

    def conjugate(self)->'ScaledIdentityMatrix':
        return ScaledIdentityMatrix(self.size, np.conjugate(self.coeff))
    
    conj = conjugate

    def __mul__(self, other)->'ScaledIdentityMatrix':
        if type(other) in [complex, float, np.float64, np.complex128]:
            return ScaledIdentityMatrix(self.size, self.coeff*other)
        else:
            return NotImplemented
    
    __rmul__ = __mul__
 
    def __matmul__(self, other: Union[MatrixBase, np.ndarray])->Union[MatrixBase,np.ndarray]:
        assert self.shape[1] == other.shape[0], f"{self.shape} {other.shape}"
        assert isinstance(other, MatrixBase) or (isinstance(other, np.ndarray) and other.ndim==1)
        return self.coeff * other

    def __add__(self, other)->MatrixBase:
        assert isinstance(other, MatrixBase)
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


class DiagonalMatrix(MatrixBase):
    """
    Diagonal matrix
    """
    def __init__(self, diagonals)->None:
        assert diagonals.ndim == 1
        self._diagonals = diagonals
        self.ndim = 2
        self.shape = (diagonals.size, diagonals.size)

    def hash(self) -> int:
        return matrix_hash(self.diagonals)
    
    @property
    def diagonals(self) -> np.ndarray:
        return self._diagonals
    
    def inv(self) -> 'DiagonalMatrix':
        return DiagonalMatrix(1/self.diagonals)

    def asmatrix(self) -> np.ndarray:
        return np.diag(self._diagonals)

    @property
    def T(self) -> 'DiagonalMatrix':
        return self
    
    def conjugate(self) -> 'DiagonalMatrix':
        return DiagonalMatrix(self.diagonals.conjugate())
    
    conj = conjugate

    def __mul__(self, other) -> 'DiagonalMatrix':
        if type(other) in [complex, float, np.float64, np.complex128]:
            return DiagonalMatrix(self._diagonals * other)
        else:
            return NotImplemented
        
    __rmul__ = __mul__

    def __add__(self, other) -> MatrixBase:
        assert isinstance(other, MatrixBase)
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


    def __matmul__(self, other: Union[MatrixBase, np.ndarray]) -> Union[MatrixBase, np.ndarray]:
        """ self @ other """
        assert self.shape[1] == other.shape[0]
        assert isinstance(other, MatrixBase) or (isinstance(other, np.ndarray) and other.ndim==1)

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

    def __str__(self) -> str:
        return "DiagonalMatrix: " + self.diagonals.__str__()


class PartialDiagonalMatrix(MatrixBase):
    """
    Matrix that can be composed as
        A otimes I.
    """
    def __init__(self, matrix: Union[np.ndarray, MatrixBase], rest_dims: tuple) -> None:
        assert matrix.ndim == 2
        self.matrix = asmatrixtype(matrix)
        self._matrix_cg = matrix.T.conj()
        self.rest_dims = rest_dims
        self.ndim = 2
        self.shape = (matrix.shape[0]*np.prod(rest_dims), matrix.shape[1]*np.prod(rest_dims))

    def hash(self) -> int:
        return matrix_hash(self.matrix)
    
    def asmatrix(self) -> np.ndarray:
        return np.einsum(
            'IJ,ij->IiJj',
            self.matrix.asmatrix(),
            np.identity(np.prod(self.rest_dims)),
            optimize=True
        ).reshape(self.shape)
    
    def inv(self) -> 'PartialDiagonalMatrix':
        return PartialDiagonalMatrix(self.matrix.inv(), self.rest_dims)

    @property
    def T(self) -> 'PartialDiagonalMatrix':
        return PartialDiagonalMatrix(self.matrix.T, self.rest_dims)

    def conjugate(self) -> 'PartialDiagonalMatrix':
        return PartialDiagonalMatrix(self.matrix.conjugate(), self.rest_dims)

    conj = conjugate

    def __matmul__(self, other) -> Union[np.ndarray, MatrixBase]:
        """ self @ other """
        assert self.shape[1] == other.shape[0]
        assert isinstance(other, MatrixBase) or isinstance(other, np.ndarray)
        if isinstance(other, np.ndarray):
            return self.matvec(other)
        elif isinstance(other, PartialDiagonalMatrix) and self.rest_dims == other.rest_dims:
            return PartialDiagonalMatrix(self.matrix@other.matrix, self.rest_dims)
        else:
            return DenseMatrix(self.asmatrix() @ other.asmatrix())


    def __mul__(self, other) -> 'PartialDiagonalMatrix':
        if np.isscalar(other):
            return PartialDiagonalMatrix(self.matrix * other, self.rest_dims)
        else:
            return NotImplemented

    __rmul__ = __mul__
    
    def __add__(self, other) -> MatrixBase:
        assert isinstance(other, MatrixBase)
        assert self.shape == other.shape

        if isinstance(other, ScaledIdentityMatrix):
            return PartialDiagonalMatrix(
                self.matrix + ScaledIdentityMatrix(self.matrix.shape[0], other.coeff), self.rest_dims)
        else:
            return DenseMatrix(self.asmatrix() + other.asmatrix())

    
    def matvec(self, v: np.ndarray) -> np.ndarray:
        """
        (a \otimes I) @ v
        v can be a vector or a tensor.
        In the latter case, the matrix applied to the first axis of v.
        """
        return _matvec_impl(self.matrix, v, self.rest_dims)


def _matvec_impl(
        matrix: Union[MatrixBase, np.ndarray],
        v: np.ndarray,
        rest_dims: tuple
    ) -> np.ndarray:
    v = v.reshape(matrix.shape[1], *rest_dims, -1)
    if isinstance(matrix, DiagonalMatrix):
        return np.einsum('d,d...->d...', matrix.diagonals, v).ravel()
    elif isinstance(matrix, DenseMatrix):
        return np.tensordot(
            matrix.asmatrix(),
            v,
            axes=(-1,0)
        ).ravel()
    else:
        raise RuntimeError(f"Unsupported type{type(matrix)}!")

def identity(n, dtype=np.float64) -> ScaledIdentityMatrix:
    """ Create an identity matrix """
    return ScaledIdentityMatrix(n, dtype(1.0))


def matrix_hash(a) -> int:
    """ Compute hash of a matrix a"""
    if isinstance(a, np.ndarray):
        return hash(a.data.tobytes()) # This makes a copy
    elif np.isscalar(a):
        return hash(a)
    else:
        return a.hash()

def asmatrixtype(a):
    assert isinstance(a, MatrixBase) or (isinstance(a, np.ndarray) and a.ndim==2)
    if isinstance(a, np.ndarray):
        return DenseMatrix(a)
    return a