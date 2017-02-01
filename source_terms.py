import numpy as np

class CenteredSourceTerm:
    def __init__(self, model):
        self.model = model

    def __call__(self, u, t):
        return self.model.source(u, t)


class BalancedSourceTerm:
    def __init__(self, grid, model, equilibrium):
        self.grid = grid
        self.model = model
        self.equilibrium = equilibrium

    def __call__(self, u, t):
        w_ref = self.equilibrium.point_values(u)
        x_ref = self.grid.cell_centers[...,0]
        x_upper = self.grid.edges[1:,...,0]
        x_lower = self.grid.edges[:-1,...,0]

        p_upper = self.equilibrium.equilibrium_values(w_ref, x_ref, x_upper)[3,...]
        p_lower = self.equilibrium.equilibrium_values(w_ref, x_ref, x_lower)[3,...]

        dudt = np.zeros_like(u)
        dudt[1,...] = (p_upper - p_lower)/self.grid.dx
        dudt[3,...] = 0.0

        return dudt
