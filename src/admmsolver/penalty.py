import numpy as np


class PenaltyBase(object):
    """
    Penalty term P(x)
    """
    def __init__(self):
        super().__init__()
    
    def solve(self, h, mu):
        """
        Return new x
            x <- argmin_x P(x) + h^+ x + x^+ h + mu x^+ x
        """
        pass


class L1Regularizer(PenaltyBase):
    """
    L1 regularization
        P(x) = alpha * |x|_1
    """
    def __init__(self, alpha):
        super().__init__()
        assert alpha > 0
        self._alpha = alpha

    def solve(self, h, mu):
        """
        x <- argmin_x alpha * |x|_1 + h^+ x + x^+ h + mu x^+ x

        This make sense only if h and x are real vectors.
        Thus, this function returns a real vector.
        """
        if np.iscomplexobj(h):
            h = h.real
        return _softmax(-h/mu, 0.5*self._alpha/mu)


def _softmax(y, lambda_):
    """
    Softmax function

    y - lambda_ (y >  lambda_)
    y + lambda_ (y < -lambda_)
    0           (otherwise)
    """
    assert lambda_ > 0
    res = np.zeros(y.size)

    idx = y > lambda_
    res[idx] = y[idx] - lambda_

    idx = y < -lambda_
    res[idx] = y[idx] + lambda_

    return res


        

        

        
