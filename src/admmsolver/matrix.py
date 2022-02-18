# Copyright (C) 2021-2022 H. Shinaoka and others
# SPDX-License-Identifier: MIT

import numpy as np
from typing import Union, Tuple, Optional, cast
from abc import abstractmethod

class MatrixBase(object):
    def __init__(self) -> None:
        super().__init__()

        self.shape = (1,1)
        self.ndim = 2

    def is_diagonal(self) -> bool:
        return self.shape[0] == self.shape[1]

    def __neg__(self)->'MatrixBase':
        return -1.0 * self

    @abstractmethod
    def asmatrix(self)->np.ndarray:
        pass

    @abstractmethod
    def __mul__(self, other: Union[float, complex, np.float64, np.complex128])->'MatrixBase':
        return NotImplemented

    __rmul__ = __mul__

    @abstractmethod
    def __matmul__(self, other: Union['MatrixBase', np.ndarray])->Union['MatrixBase',np.ndarray]:
        return NotImplemented

    @abstractmethod
    def __add__(self, other: 'MatrixBase')->'MatrixBase':
        return NotImplemented

    @abstractmethod
    def conjugate(self)->'MatrixBase':
        pass

    @property
    @abstractmethod
    def T(self)->'MatrixBase':
        pass

    conj = conjugate

    def __sub__(self, other)->'MatrixBase':
        return self + (-other)

    @abstractmethod
    def inv(self)->'MatrixBase':
        pass

    @abstractmethod
    def hash(self)->int:
        pass

class DenseMatrix(MatrixBase):
    def __init__(self, matrix: np.ndarray) -> None:
        assert isinstance(matrix, np.ndarray)
        assert matrix.ndim == 2
        self.data = matrix
        self.shape = cast(Tuple[int,int], matrix.shape)
        self.ndim = 2

    def hash(self)->int:
        return matrix_hash(self.asmatrix())

    def asmatrix(self)->np.ndarray:
        return self.data

    def inv(self)->'DenseMatrix':
        return DenseMatrix(np.linalg.inv(self.asmatrix()))

    def __neg__(self)->'DenseMatrix':
        return -1.0 * self

    @property
    def T(self):
        return DenseMatrix(self.data.T)

    def conjugate(self):
        return DenseMatrix(self.data.conjugate())

    conj = conjugate

    def __mul__(self, other)->'DenseMatrix':
        if np.isscalar(other):
            return DenseMatrix(self.data * other)
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
                return self @ other.to_diagonal_matrix()
            elif isinstance(other, DiagonalMatrix):
                res = np.zeros((self.shape[0], other.shape[1]), dtype=np.complex128) # make real if possible
                min_size = min(*other.shape)
                res[:, 0:min_size] = \
                    self.data[:, 0:min_size] * other.diagonals[None,:]
                return DenseMatrix(res)
            else:
                return DenseMatrix(self.data @ other.asmatrix())

    def __add__(self, other)->'DenseMatrix':
        assert isinstance(other, MatrixBase)
        assert self.shape == other.shape

        return DenseMatrix(self.asmatrix() + other.asmatrix())



class ScaledIdentityMatrix(MatrixBase):
    """
    Scaled Identity matrix with a rectangular shape
    """
    def __init__(
            self,
            shape: Union[int, Tuple[int,int]],
            coeff: Union[complex, float, np.float64, np.complex128]
        )->None:
        assert type(coeff) in [complex, float, np.float64, np.complex128], type(coeff)
        self.shape = (0,0) # type: Tuple[int, int]
        if isinstance(shape, int):
            self.shape = (shape, shape)
        elif isinstance(shape, tuple):
            self.shape = shape
        else:
            raise ValueError("Invalid shape value!")
        self.coeff = coeff # type: Union[complex, float, np.float64, np.complex128]
        self.ndim = 2 # type: int

    def hash(self)->int:
        return matrix_hash(self.coeff)

    def asmatrix(self) -> np.ndarray:
        return self.coeff * np.eye(N=self.shape[0], M=self.shape[1])

    def __neg__(self)->'ScaledIdentityMatrix':
        return -1.0 * self

    def inv(self)->'ScaledIdentityMatrix':
        if not self.is_diagonal():
            raise RuntimeError("A rectangular matrix is not invertible!")
        return ScaledIdentityMatrix(self.shape, 1/self.coeff)

    @property
    def T(self) -> 'ScaledIdentityMatrix':
        return ScaledIdentityMatrix((self.shape[1], self.shape[0]), self.coeff)

    @property
    def diagonals(self)->np.ndarray:
        if not self.is_diagonal():
            raise RuntimeError("Diagonals of a rectangular matrix is ill defined!")
        return np.full(self.shape[0], self.coeff)

    def conjugate(self)->'ScaledIdentityMatrix':
        return ScaledIdentityMatrix(self.shape, np.conjugate(self.coeff))

    conj = conjugate

    def __mul__(self, other)->'ScaledIdentityMatrix':
        if type(other) in [complex, float, np.float64, np.complex128]:
            return ScaledIdentityMatrix(self.shape, self.coeff*other)
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __matmul__(self, other: Union[MatrixBase, np.ndarray])->Union[MatrixBase,np.ndarray]:
        assert self.shape[1] == other.shape[0], f"{self.shape} {other.shape}"
        assert isinstance(other, MatrixBase) or isinstance(other, np.ndarray)
        return self.to_diagonal_matrix() @ other

    def __add__(self, other: MatrixBase)->MatrixBase:
        assert isinstance(other, MatrixBase)
        assert self.shape == other.shape, f"{self.shape} {other.shape}"

        if isinstance(other, ScaledIdentityMatrix):
            return ScaledIdentityMatrix(self.shape, self.coeff + other.coeff)
        elif isinstance(other, DiagonalMatrix):
            return DiagonalMatrix(self.coeff*np.ones(self.shape[0]) + other.diagonals)
        elif isinstance(other, PartialDiagonalMatrix):
            return PartialDiagonalMatrix(
                ScaledIdentityMatrix(other.matrix.shape[0], self.coeff) + other.matrix,
                other.rest_dims)
        else:
            return DenseMatrix(self.asmatrix() + other.asmatrix())

    def to_diagonal_matrix(self):
        """ Convert to a diagonal matrix """
        return DiagonalMatrix(self.coeff*np.ones(min(*self.shape)), self.shape)

class DiagonalMatrix(MatrixBase):
    """
    Diagonal matrix
    """
    def __init__(self, diagonals, shape: Optional[Tuple[int,int]] = None)->None:
        assert diagonals.ndim == 1
        self._diagonals = diagonals
        self.ndim = 2
        if shape is None:
            self.shape = (diagonals.size, diagonals.size)
        else:
            self.shape = shape
        assert min(*self.shape) == diagonals.size, f"{self.shape} {diagonals.size}"

    def hash(self) -> int:
        return matrix_hash(self.diagonals)

    @property
    def diagonals(self) -> np.ndarray:
        return self._diagonals

    def __neg__(self)->'DiagonalMatrix':
        return -1.0 * self

    def inv(self) -> 'DiagonalMatrix':
        if not self.is_diagonal():
            raise RuntimeError("Must be a diagonal matrix!")
        return DiagonalMatrix(1/self.diagonals)

    def asmatrix(self) -> np.ndarray:
        mat = np.zeros(self.shape, dtype=self.diagonals.dtype)
        min_size = min(*self.shape)
        for i in range(min_size):
            mat[i,i] = self.diagonals[i]
        return mat

    @property
    def T(self) -> 'DiagonalMatrix':
        return DiagonalMatrix(self.diagonals, shape=(self.shape[1], self.shape[0]))

    def conjugate(self) -> 'DiagonalMatrix':
        return DiagonalMatrix(self.diagonals.conjugate(), self.shape)

    conj = conjugate

    def __mul__(self, other) -> 'DiagonalMatrix':
        if type(other) in [complex, float, np.float64, np.complex128]:
            return DiagonalMatrix(self._diagonals * other, self.shape)
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
        assert isinstance(other, MatrixBase) or isinstance(other, np.ndarray)

        if isinstance(other, np.ndarray):
            dtype = np.dtype(type(self._diagonals[0] * other.ravel()[0]))
            if other.ndim == 1:
                res = np.zeros(self.shape[0], dtype=dtype)
                min_len = min(self._diagonals.size, other.size)
                res[0:min_len] = self._diagonals[0:min_len] * other[0:min_len]
                return res
            else:
                rest_dim = other.shape[1:]
                res = np.zeros((self.shape[0],) + rest_dim, dtype=dtype)
                min_len = min(self._diagonals.size, other.size)
                res[0:min_len, ...] = \
                    np.einsum('d,d...->d...', self._diagonals[0:min_len], other[0:min_len, ...], optimize=True)
                return res
        elif isinstance(other, DenseMatrix):
            return DenseMatrix(self._diagonals[:,None] * other.data)
        elif isinstance(other, DiagonalMatrix):
            min_size = min(self.shape[0], other.shape[1])
            return DiagonalMatrix(
                _vecprod(self._diagonals, other._diagonals, min_size),
                (self.shape[0], other.shape[1])
            )
        elif isinstance(other, PartialDiagonalMatrix):
            return DenseMatrix(self._diagonals[:,None] * other.asmatrix())
        elif isinstance(other, ScaledIdentityMatrix):
            return self @ other.to_diagonal_matrix()
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

    def __neg__(self)->'PartialDiagonalMatrix':
        return -1.0 * self

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
        if type(other) in [float, complex, np.float64, np.complex128]:
            return PartialDiagonalMatrix(self.matrix * other, self.rest_dims)
        else:
            return NotImplemented

    __rmul__ = __mul__

    def __add__(self, other) -> MatrixBase:
        assert isinstance(other, MatrixBase)
        assert self.shape == other.shape

        if isinstance(other, ScaledIdentityMatrix):
            return PartialDiagonalMatrix(
                    self.matrix + ScaledIdentityMatrix(self.matrix.shape, other.coeff),
                    self.rest_dims
                )
        else:
            return DenseMatrix(self.asmatrix() + other.asmatrix())


    def matvec(self, v: np.ndarray) -> np.ndarray:
        r"""
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

    res_leading_dim = matrix.shape[0] * np.prod(rest_dims)
    res_shape = (0,) # type: Tuple[int,...]
    if v.ndim == 1:
        res_shape = (res_leading_dim,)
    else:
        res_shape = (res_leading_dim,) + v.shape[1:]

    v = v.reshape(matrix.shape[1], *rest_dims, -1)
    if isinstance(matrix, DiagonalMatrix):
        return cast(np.ndarray, matrix @ v).reshape(res_shape)
    elif isinstance(matrix, DenseMatrix):
        return np.tensordot(
            matrix.asmatrix(),
            v,
            axes=(-1,0)
        ).reshape(res_shape)
    elif isinstance(matrix, ScaledIdentityMatrix):
        return (matrix.coeff * v.ravel()).reshape(res_shape)
    else:
        raise RuntimeError(f"Unsupported type{type(matrix)}!")

def identity(n, dtype=np.float64) -> ScaledIdentityMatrix:
    """ Create an identity matrix """
    n = int(n)
    assert isinstance(n, int), n
    return ScaledIdentityMatrix(n, dtype(1.0))


def matrix_hash(a) -> int:
    """ Compute hash of a matrix a"""
    if isinstance(a, np.ndarray):
        return hash(a.data.tobytes()) # This makes a copy
    elif np.isscalar(a):
        return hash(a)
    else:
        return a.hash()

def asmatrixtype(a) -> MatrixBase:
    assert isinstance(a, MatrixBase) or (isinstance(a, np.ndarray) and a.ndim==2)
    if isinstance(a, np.ndarray):
        return DenseMatrix(a)
    return a


def _vecprod(v1: np.ndarray, v2: np.ndarray, size=Optional[int]):
    """
    Elementwise product of two vectors.
    If v1.size < size and v2.size, the result is padded on the right
    so that the returned array is size.
    """
    assert isinstance(v1, np.ndarray)
    assert isinstance(v2, np.ndarray)
    min_size = min(v1.size, v2.size)
    res = v1[0:min_size] * v2[0:min_size]
    return _pad_by_zero(res, size)

def _pad_by_zero(arr: np.ndarray, size: int):
    assert arr.size <= size
    if arr.size == size:
        return arr
    res = np.zeros(size, dtype=arr.dtype)
    res[0:arr.size] = arr
    return res