import numpy as np

class BoundaryCondition(object):
    def __init__(self, grid):
        self.grid = grid

class Periodic(BoundaryCondition):
    def __call__(self, u):
        n_ghost = self.grid.n_ghost

        u[:,:n_ghost,:] = u[:,-2*n_ghost:-n_ghost,:]
        u[:,-n_ghost:,:] = u[:,n_ghost:2*n_ghost,:]

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
