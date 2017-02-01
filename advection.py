import numpy as np

class Advection(object):
    """Simple advection equation."""

    def __init__(self, velocity):
        self.velocity = velocity

    def flux(self, u, axis):
        return self.velocity[axis]*u

    def source(self, u, t):
        return 0.0

    def max_eigenvalue(self, u):
        return np.max(np.abs(self.velocity))
