from admmsolver.util import *

def test_second_deriv_prj():
    """
    f(x) = x^2
    f''(x) = 2
    """
    xmax = 3
    N = 1000
    x = np.linspace(0, np.sqrt(xmax), N)**2 # Non-uniform mesh

    prj = second_deriv_prj(x)
    ypp = prj @ (x**2)
    ypp_ref = np.full(N-2, 2)

    np.testing.assert_allclose(ypp, ypp_ref)


def test_smooth_regularizer_coeff():
    """
    f(x) = x^2
    f''(x) = 2
    """
    omega_min = 0.0
    omega_max = 3.0
    N = 10000
    omega = np.linspace(np.sqrt(omega_min), np.sqrt(omega_max), N)**2 # Non-uniform mesh

    prj = smooth_regularizer_coeff(omega)

    ypp = 2
    assert np.abs(
           np.linalg.norm(prj @ omega**2)**2 - 
           (omega_max-omega_min) * ypp**2
        ) < 1e-2