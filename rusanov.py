import numpy as np

class Rusanov(object):
    """Rusanov's numerical flux for hyperbolic PDEs."""

    def __init__(self, model):
        """Create Rusanov's flux for `model` equations."""
        self.model = model

    def __call__(self, u_left, u_right, axis):
        c = np.maximum(self.model.max_eigenvalue(u_left),
                       self.model.max_eigenvalue(u_right))
        return 0.5*(self.model.flux(u_left, axis) + self.model.flux(u_right, axis)
                    - c*(u_right - u_left))

