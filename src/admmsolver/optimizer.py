import numpy as np

from .main_objective import MainLestSquaresBase

class PlainOptimizer:
    """
    The simplest ADMM solver
    """
    def __init__(self, main_objective, sub_objectives):
        assert isinstance(main_objective, MainLestSquaresBase)

        self._num_subobj = len(sub_objectives)

    def solve(self, x0=None, maxiter=10000, nu=None, mu=None):
        # Penalty terms
        if mu is None:
            mu = np.zeors(self._num_subobj, dtype=np.float64)
        mu = np.asarray(mu, dtype=np.float64)

        if nu is None:
            nu = [
                np.zeors(self._Nx, dtype=np.complex128)
                for i in range(self._num_subobj)
            ]
        
        if x0 is None:
            x = [
                np.zeors(self._Nx, dtype=np.complex128)
                for i in range(self._num_subobj+1)
            ]
        
        for iter in range(maxiter):
            # Optimize x and nu

            # Optimize x_i

            # Optimize h_i

            pass

        return x # Return all x's