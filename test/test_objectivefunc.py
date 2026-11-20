import numpy as np
from scipy.optimize import minimize
from typing import cast

from admmsolver.objectivefunc import *
from admmsolver.matrix import identity, asmatrixtype, PartialDiagonalMatrix

import pytest

def _randn_cmplx(*shape) -> np.ndarray:
    return np.random.randn(*shape) + 1j* np.random.randn(*shape)

def _to_real_array(x):
    if np.iscomplexobj(x):
        return x.view(np.float64).ravel()
    else:
        return x

def _from_real_array(x):
    x = x.reshape((-1, 2))
    x = x.view(np.complex128)
    return x.reshape(x.shape[0:-1])

def _minimize(f, x0, method="BFGS"):
    x0 = _to_real_array(x0)
    res = minimize(
            lambda x: f(_from_real_array(x)),
            x0,
            method=method,
            options={'maxiter': 100000}
        )
    return _from_real_array(res.x)

def test_least_squares():
    """
    Least-squares fit with a dense coefficient matrix
    """
    np.random.seed(100)
    N1, N2 = 4, 2
    alpha = 2.0
    y = _randn_cmplx(N1)
    A = _randn_cmplx(N1, N2)
    h = _randn_cmplx(N2)
    sqrt_mu = _randn_cmplx(N2, N2)
    mu = asmatrixtype(sqrt_mu.T.conjugate() @ sqrt_mu)
    lstsq = LeastSquares(alpha, A, y)
    x = lstsq.solve(h, mu)

    f_all = lambda x: np.real(alpha * np.linalg.norm(y - A @ x)**2 + \
        h.T.conjugate() @ x + x.T.conjugate() @ h + x.conjugate().T @ (mu @ x))
    x_ref = _minimize(f_all, x, "BFGS")
    np.testing.assert_allclose(x, x_ref, rtol=1e-8)
    np.testing.assert_allclose(f_all(x), f_all(x_ref), rtol=1e-8)


def test_least_squares_partial():
    """
    Least-squares fit with a coefficient matrix of type PartialDiagonalMatrix
    """
    np.random.seed(100)
    N1, N2 = 40, 20
    alpha = 2.0
    rest_dim_size = 20
    y = _randn_cmplx(N1)
    A = PartialDiagonalMatrix(_randn_cmplx(N1//rest_dim_size, N2//rest_dim_size), rest_dims=(rest_dim_size,))
    assert A.shape == (N1, N2)
    h = _randn_cmplx(N2)
    sqrt_mu = _randn_cmplx(N2, N2)
    mu = asmatrixtype(sqrt_mu.T.conjugate() @ sqrt_mu)
    lstsq = LeastSquares(alpha, A, y)
    x = lstsq.solve(h, mu)

    f_all = lambda x: \
        np.real(
            alpha * np.linalg.norm(y - A @ x)**2 + h.T.conjugate() @ x + \
                x.T.conjugate() @ h + x.conjugate().T @ (mu @ x))
    x_ref = _minimize(f_all, x, "BFGS")
    np.testing.assert_allclose(x, x_ref, rtol=1e-4)
    np.testing.assert_allclose(f_all(x), f_all(x_ref), rtol=1e-8)


def test_constrained_least_squares():
    """
    Linearly constrained least squares
    """
    np.random.seed(100)

    N1, N2 = 8, 4
    Nc = 2
    alpha = 2.0
    y = _randn_cmplx(N1)
    A = _randn_cmplx(N1, N2)
    h = _randn_cmplx(N2)
    C = _randn_cmplx(Nc, N2)
    D = _randn_cmplx(Nc)
    sqrt_mu = _randn_cmplx(N2, N2)
    mu = asmatrixtype(sqrt_mu.T.conjugate() @ sqrt_mu)
    lstsq = ConstrainedLeastSquares(alpha, A, y, C, D)

    x = lstsq.solve(h, mu)
    assert np.abs(C@x - D).max() < 1e-10

    #FIXME: how to check x?


def test_L1():
    """
    Minimize alpha * |x|_1 + h^+ x + x^+ h + mu x^+ x
    """
    N = 20
    assert N%2 == 0
    h = 0.5*np.arange(-N//2, N//2)
    mu = identity(N)
    alpha = 1.0

    l1 = L1Regularizer(alpha, N)
    x = l1.solve(h, mu)

    # Naive optimization
    for i in range(N):
        f = lambda x: alpha * np.abs(x) + 2*h[i]*x + mu.diagonals[i] * x**2
        x0 = 0.0
        res = minimize(f, x0, method="BFGS")
        assert np.abs(x[i]-res.x[0]) < 1e-5

def test_non_negative():
    """
    Minimize infty * Theta(-x) + h^+ x + x^+ h + mu x^+ x
    """
    h = np.array([0, -10, 10])
    N = h.size
    mu = identity(N)

    func = NonNegativePenalty(N)
    x = func.solve(h, mu)

    step_f = lambda x: x if x >= 0 else 0

    # Naive optimization
    for i in range(N):
        f = lambda x: 1e+5 * step_f(-x) + 2*h[i]*x + mu.diagonals[i] * x**2
        x0 = 0.0
        res = minimize(f, x0, method="BFGS")
        assert np.abs(x[i]-res.x[0]) < 1e-5


def test_L2():
    """
    Minimize alpha * |A x|_2^2 + h^+ x + x^+ h + mu x^+ x
    """
    N = 10
    M = 5
    sqrt_mu = _randn_cmplx(N, N)
    mu = asmatrixtype(sqrt_mu.T.conjugate() @ sqrt_mu)
    alpha = 2.0
    A = _randn_cmplx(M, N)
    h = _randn_cmplx(N)

    l2 = L2Regularizer(alpha, A)
    x = l2.solve(h, mu)

    # Naive optimization
    f = lambda x: alpha * np.linalg.norm(A @ x)**2 + \
        2*np.real(h.conjugate().T @ x) + np.real(x.conjugate().T @ (mu @ x))
    x_ref = _minimize(f, x, method="BFGS")
    np.testing.assert_allclose(cast(np.ndarray, x), x_ref, atol=np.abs(x_ref).max()*1e-5, rtol=0)


def test_semi_positive_definite_penalty():
    np.random.seed(100)
    K = 20
    N = 10

    h = _randn_cmplx(N**2 * K)

    for mu in [
        asmatrixtype(identity(N**2 * K)),
        PartialDiagonalMatrix(ScaledIdentityMatrix(N**2, 1.0), (K,))]:
        p = SemiPositiveDefinitePenalty((N,N,K), axis=2)
        res = p.solve(h, mu)

        x = res.reshape((N,N,K))
        for k in range(K):
            evals, evecs = np.linalg.eigh(x[:,:,k])
            assert all(evals > -1e-10)



def _old_softmax(y, lambda_):
    """Reference: original real-only piecewise soft-threshold."""
    res = np.zeros(y.size)
    idx = y > lambda_
    res[idx] = y[idx] - lambda_[idx]
    idx = y < -lambda_
    res[idx] = y[idx] + lambda_[idx]
    return res


def test_softmax_real_equivalence_and_dtype():
    from admmsolver.objectivefunc import _softmax
    rng = np.random.RandomState(0)
    y = rng.randn(200)
    lam = 0.3 * np.ones_like(y)
    new = _softmax(y, lam)
    old = _old_softmax(y, lam)
    assert np.array_equal(new, old)                      # exact, not just close
    assert new.dtype == np.float64                       # real stays float64
    # exact boundaries: 0, +-lambda, and nextafter on both sides
    lm = 1.5
    ys = np.array([0.0, lm, -lm,
                   np.nextafter(lm, np.inf), np.nextafter(lm, -np.inf),
                   np.nextafter(-lm, np.inf), np.nextafter(-lm, -np.inf),
                   1e12, -1e12], dtype=np.float64)
    lams = np.full(ys.size, lm)
    assert np.array_equal(_softmax(ys, lams), _old_softmax(ys, lams))
    # float32 input is promoted to float64 (no threshold truncation for ints)
    assert _softmax(np.array([5, -5, 0]), np.full(3, 1.5)).dtype == np.float64
    assert np.array_equal(_softmax(np.array([5, -5, 0]), np.full(3, 1.5)), [3.5, -3.5, 0.0])
    assert _softmax(np.array([1.0], np.float32), np.array([0.3])).dtype == np.float64
    # real +-inf stays +-inf (not NaN)
    r = _softmax(np.array([np.inf, -np.inf]), np.array([1.0, 1.0]))
    assert r[0] == np.inf and r[1] == -np.inf


def test_softmax_complex_and_inf():
    from admmsolver.objectivefunc import _softmax
    # complex phase preserved, magnitude shrunk
    y = np.array([3.0 + 4.0j, 0.6 - 0.8j, 0.0 + 0.0j])   # |y| = 5, 1, 0
    lam = np.array([1.0, 2.0, 1.0])
    out = _softmax(y, lam)
    assert out.dtype == np.complex128
    assert np.allclose(out[0], (5.0 - 1.0) * (y[0] / 5.0))   # shrink 5 -> 4, keep phase
    assert out[1] == 0                                        # |y|=1 < lambda=2
    assert out[2] == 0                                        # zero input
    # complex entries with an infinite component must not become NaN
    yi = np.array([complex(np.inf, 1.0), complex(1.0, np.inf), complex(np.inf, np.inf)])
    ri = _softmax(yi, np.ones(3))
    assert not np.isnan(ri).any()
    assert ri[0] == complex(np.inf, 1.0) and ri[1] == complex(1.0, np.inf)


def test_L1_complex():
    """Complex L1 prox = magnitude soft-threshold keeping the phase."""
    N = 30
    rng = np.random.RandomState(1)
    h = rng.randn(N) + 1j * rng.randn(N)
    d = 0.5 + np.arange(N) * 0.1          # nonuniform positive weights
    mu = DiagonalMatrix(d)
    alpha = 0.7
    x = L1Regularizer(alpha, N).solve(h, mu)
    assert x.dtype == np.complex128
    # closed form
    center = -h / d
    lam = 0.5 * alpha / d
    mag = np.abs(center)
    expect = np.where(mag > 0, np.maximum(mag - lam, 0.0) * center / np.where(mag > 0, mag, 1.0), 0.0)
    assert np.allclose(x, expect, atol=1e-12)
    # objective optimality (per component): f(x) <= f(x + perturbation)
    for i in range(0, N, 7):
        f = lambda z: alpha * np.abs(z) + 2 * np.real(np.conj(h[i]) * z) + d[i] * np.abs(z) ** 2
        for pert in [0.05, -0.05, 0.05j, -0.05j, 0.03 + 0.03j]:
            assert f(x[i]) <= f(x[i] + pert) + 1e-12


def test_spd_penalty_complex_correctness():
    """Complex-Hermitian SPD projection with a scalar (uniform) penalty must
    equal the eigenvalue-clipped Hermitization of the unconstrained centre."""
    rng = np.random.RandomState(2)
    K, N = 6, 3
    h = rng.randn(N * N * K) + 1j * rng.randn(N * N * K)
    d = 2.0
    mu = PartialDiagonalMatrix(ScaledIdentityMatrix(N * N, d), (K,))
    res = SemiPositiveDefinitePenalty((N, N, K), axis=2).solve(h, mu).reshape((N, N, K))
    hb = (-h / d).reshape((N, N, K))
    for k in range(K):
        H = 0.5 * (hb[:, :, k] + hb[:, :, k].conj().T)      # Hermitize
        ev, U = np.linalg.eigh(H)
        keep = ev >= 0
        ref = U[:, keep] @ (ev[keep][:, None] * U[:, keep].conj().T)
        assert np.allclose(res[:, :, k], ref, atol=1e-10)          # exact projection
        assert np.allclose(res[:, :, k], res[:, :, k].conj().T, atol=1e-12)  # Hermitian
        assert np.linalg.eigvalsh(res[:, :, k]).min() > -1e-10              # PSD
    # imaginary part is genuinely retained (not dropped)
    assert np.max(np.abs(res.imag)) > 1e-3


def test_spd_penalty_nonuniform_center():
    """Verify the weighted paired-Hermitian center for a non-uniform diagonal
    metric (independent of the known non-uniform PSD-projection limitation).
    Uses a case whose center is already PSD, so eigenvalue clipping is a no-op
    and the result must equal the center exactly."""
    N, K = 2, 1
    # per-entry weights d (Hermitian-pattern positive); build via DiagonalMatrix
    d = np.array([1.0, 3.0, 0.5, 2.0])            # (0,0),(0,1),(1,0),(1,1)
    # choose h so that centre = -(h_ij + conj(h_ji))/(d_ij + d_ji) is PSD
    h = np.array([-2.0 + 0j, -(0.3 + 0.2j), -(0.3 - 0.2j), -3.0 + 0j])
    mu = DiagonalMatrix(d)
    res = SemiPositiveDefinitePenalty((N, N, K), axis=2).solve(h, mu).reshape((N, N))
    hb = h.reshape((N, N)); db = d.reshape((N, N))
    centre = -(hb + hb.conj().T) / (db + db.T)
    assert np.linalg.eigvalsh(centre).min() >= 0            # center already PSD
    assert np.allclose(res, centre, atol=1e-12)             # clip is a no-op -> equals center
    assert np.allclose(res, res.conj().T)                   # Hermitian


def test_spd_penalty_uniform_diagonal_matches_scaled_identity():
    """A uniform full DiagonalMatrix must give the same result as the scalar
    ScaledIdentityMatrix path."""
    rng = np.random.RandomState(3)
    N, K = 3, 4
    h = rng.randn(N * N * K) + 1j * rng.randn(N * N * K)
    c = 1.7
    p = SemiPositiveDefinitePenalty((N, N, K), axis=2)
    r_scaled = p.solve(h, PartialDiagonalMatrix(ScaledIdentityMatrix(N * N, c), (K,)))
    r_diag = p.solve(h, DiagonalMatrix(np.full(N * N * K, c)))
    assert np.allclose(r_scaled, r_diag, atol=1e-10)


def test_softmax_low_precision_bit_exact():
    """Real path must be bit-for-bit identical to the historical implementation
    for every real dtype (arithmetic in operand dtype, float64 result)."""
    from admmsolver.objectivefunc import _softmax
    rng = np.random.RandomState(4)
    for dt in (np.float16, np.float32, np.float64):
        y = (rng.randn(64) * 3).astype(dt)
        lam = np.abs(rng.randn(64)).astype(dt) + dt(0.1)
        assert np.array_equal(_softmax(y, lam), _old_softmax(y, lam))
    # integer input: no threshold truncation, float64 output
    yi = np.array([5, -5, 0, 2, -2])
    li = np.full(yi.size, 1.5)
    assert np.array_equal(_softmax(yi, li), _old_softmax(yi, li))
    assert _softmax(yi, li).dtype == np.float64


def test_spd_center_no_overflow_large_values():
    """Paired centre must stay finite for near-float64-max operands."""
    big = np.finfo(np.float64).max
    N, K = 2, 1
    # uniform h = d = big  ->  centre_ij = -(big+big)/(big+big) = -1 (must not be NaN)
    h = np.array([big, big, big, big], dtype=np.float64)
    d = np.array([big, big, big, big], dtype=np.float64)
    p = SemiPositiveDefinitePenalty((N, N, K), axis=2)
    res = p.solve(h, DiagonalMatrix(d)).reshape((N, N))
    assert np.isfinite(res).all()
    centre = -np.ones((N, N))                       # all entries -1
    ev, U = np.linalg.eigh(centre)
    ref = U[:, ev >= 0] @ (ev[ev >= 0][:, None] * U[:, ev >= 0].conj().T)
    assert np.allclose(res, ref, atol=1e-8)


def test_softmax_large_finite_complex_no_overflow():
    """Large-but-finite complex magnitude must be shrunk (not passed through),
    even when |y| = sqrt(re^2+im^2) overflows float64."""
    from admmsolver.objectivefunc import _softmax
    y = np.array([1.3e308 + 1.3e308j])
    lam = np.array([1.0e308])
    out = _softmax(y, lam)
    assert np.isfinite(out).all()
    # exact prox = (1 - lambda/|y|) * y, with lambda/|y| via component scaling so
    # the reference itself does not overflow ( |y| ~ 1.84e308 > DBL_MAX ).
    m = max(abs(y[0].real), abs(y[0].imag))
    ratio = (lam[0] / m) / np.hypot(y[0].real / m, y[0].imag / m)
    expect = (1.0 - ratio) * y[0]
    assert np.isclose(out[0].real, expect.real, rtol=1e-12)
    assert np.isclose(out[0].imag, expect.imag, rtol=1e-12)
    assert abs(out[0].real) < 6.0e307        # genuinely shrunk from 1.3e308


def test_spd_center_large_finite_complex():
    """SPD paired centre must stay finite for near-max complex operands."""
    big = np.finfo(np.float64).max
    N, K = 2, 1
    h = np.array([complex(-big, big)] * (N * N))
    d = np.array([big] * (N * N))
    res = SemiPositiveDefinitePenalty((N, N, K), axis=2).solve(h, DiagonalMatrix(d)).reshape((N, N))
    assert np.isfinite(res).all()
    # centre_ij = -(h_ij + conj(h_ji))/(d_ij+d_ji) = -(-big+big j + (-big-big j))/(2 big) = 1
    ones = np.ones((N, N))
    assert np.allclose(res, ones, atol=1e-8)      # ones is PSD -> projection is itself
