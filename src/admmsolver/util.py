import numpy as np
from typing import Callable, cast

def second_deriv_prj(x):
    """
    x: 1D array of foats in strictly increasing order

    Return a dense matrix that approximates the second derivative of a given y(x) as
    y''(x_i) simeq = sum_j P_{ij} x_j
    for i=1, 2, ..., N-1.
    """
    assert all(x[1:] > x[0:-1]), "x must be in increasing order!"

    prj = np.zeros((x.size-2, x.size), dtype=np.float64)
    for i in range(x.size-2):
        ip = i + 1
        dx_forward = x[ip+1] - x[ip]
        dx_backward = x[ip] - x[ip-1]
        coeff = 2/(dx_forward**2 * dx_backward + dx_backward**2 * dx_forward)
        prj[i,ip-1] = coeff * dx_forward
        prj[i,ip  ] = coeff * (-dx_backward - dx_forward)
        prj[i,ip+1] = coeff * dx_backward
    return prj


def smooth_regularizer_coeff(omega):
    """
    omegqa: 1D array of foats in strictly increasing order

    Return a dense matrix that approximates the integral of the squared second derivative of y(omega) as

    int domega |y''(omega)|^2
    = \sum_{i=1}^{N-2} |sum_j P_{ij} y_j|^2,
    """
    assert all(omega[1:] > omega[0:-1]), "omega must be in increasing order!"

    dx = 0.5*(omega[2:] - omega[0:-2])
    prj_second_deriv = second_deriv_prj(omega)
    return np.sqrt(dx)[:,None] * prj_second_deriv

norm = lambda x: cast(float, np.linalg.norm(cast(np.ndarray, x))) # Callable[np.ndarray, float]