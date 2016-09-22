import numpy as np

class BoundaryCondition(object):
    def __init__(self, grid):
        self.grid = grid

class Periodic(BoundaryCondition):
    def __call__(self, u):
        n_ghost = self.grid.n_ghost

        u[:n_ghost,:,:] = u[-2*n_ghost:-n_ghost,:,:]
        u[-n_ghost:,:,:] = u[n_ghost:2*n_ghost,:,:]

        if self.grid.n_dims == 2:
            u[:,:n_ghost,:] = u[:,-2*n_ghost:-n_ghost,:]
            u[:,-n_ghost:,:] = u[:,n_ghost:2*n_ghost,:]
