import numpy as np

class Burgers(object):
    """Burgers equation is a simple non-linear PDE."""

    def flux(self, u, axis):
        return 0.5*u[axis,...]*u

    def source(self, u):
        return 0.0

    def max_eigenvalue(self, u):
        return np.abs(u)

