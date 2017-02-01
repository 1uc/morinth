import numpy as np
from equilibrium import IsothermalRC

class BoundaryCondition(object):
    def __init__(self, grid):
        self.grid = grid

class Periodic(BoundaryCondition):
    def __call__(self, u):
        n_ghost = self.grid.n_ghost

        u[:,:n_ghost,...] = u[:,-2*n_ghost:-n_ghost,...]
        u[:,-n_ghost:,...] = u[:,n_ghost:2*n_ghost,...]

        if self.grid.n_dims == 2:
            u[:,:,:n_ghost] = u[:,:,-2*n_ghost:-n_ghost]
            u[:,:,-n_ghost:] = u[:,:,n_ghost:2*n_ghost]


class Outflow(BoundaryCondition):
    def __call__(self, u):
        n_ghost = self.grid.n_ghost

        if self.grid.n_dims == 1:
            shape = (u.shape[0],) + (1,)
        else:
            shape = (u.shape[0],) + (1,) + (u.shape[2],)

        u[:,:n_ghost,...] = u[:,n_ghost,...].reshape(shape)
        u[:,-n_ghost:,...] = u[:,-n_ghost-1,...].reshape(shape)

        if self.grid.n_dims == 2:
            shape = u.shape[:2] + (1,)
            u[:,:,:n_ghost] = u[:,:,n_ghost].reshape(shape)
            u[:,:,-n_ghost:] = u[:,:,-n_ghost-1].reshape(shape)


class HydrostaticOutflow(BoundaryCondition):
    def __init__(self, grid, equilibrium):
        self.grid = grid
        self.equilibrium = equilibrium

    def __call__(self, u):
        n_ghost = self.grid.n_ghost
        self.set_layer(u, n_ghost, slice(0, n_ghost))
        self.set_layer(u, -n_ghost-1, slice(-n_ghost, None))

    def set_layer(self, u, i_inner, i_outer):
        w_ref = self.equilibrium.point_values(u[:,i_inner,...])
        x_ref = self.grid.cell_centers[i_inner,...,0]
        x = self.grid.cell_centers[i_outer,...,0]

        w = self.equilibrium.equilibrium_values(w_ref, x_ref, x)
        u[:,i_outer,...] = self.equilibrium.model.conserved_variables(w)


class IsothermalOutflow(HydrostaticOutflow):
    def __init__(self, grid, model):
        super().__init__(grid, IsothermalRC(grid, model))
