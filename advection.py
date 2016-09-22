import numpy as np

class Advection(object):
    """Simple advection equation."""

    def __init__(self, velocity):
        self.velocity = velocity

    def flux(self, u, axis):
        return self.velocity[axis]*u

    def max_eigenvalue(self, u):
        return np.max(np.abs(self.velocity))
