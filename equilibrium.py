import numpy as np

class IsothermalEquilibrium:
    def __init__(self, model, p_ref, T_ref, x_ref):
        self.model = model
        self.p_ref = p_ref
        self.T_ref = T_ref
        self.x_ref = x_ref

    def __call__(self, x):
        scale_height = self.model.scale_height(self.T_ref)

        p = self.p_ref*np.exp(-(x - self.x_ref)/scale_height)
        rho = self.model.rho(p=p, T=self.T_ref)

        return rho, p

class IsothermalRC:
    """Equilibrium reconstruction of point-values."""

    def __init__(self, grid, model):
        self.grid = grid
        self.model = model

    def point_values(self, u):
        return self.model.primitive_variables(u)

    def equilibrium_values(self, w_ref, x_ref, x, is_reversed=False):
        w_wb = np.empty((4,) + x.shape)

        p_ref = w_ref[3,...]
        T_ref = self.model.temperature(rho=w_ref[0,...], p=w_ref[3,...])

        equilibrium = IsothermalEquilibrium(self.model, p_ref, T_ref, x_ref)

        w_wb[0,...], w_wb[3,...] = equilibrium(x)

        w_wb[1,...] = w_ref[1,...]
        w_wb[2,...] = w_ref[2,...]

        return w_wb

    def delta(self, w_ref, x_ref, w, x, is_reversed):
        """Compute the difference from equilibrium."""
        dw = w - self.equilibrium_values(w_ref, x_ref, x, is_reversed)
        return dw
