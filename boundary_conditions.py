import numpy as np
from equilibrium import IsothermalEquilibrium

class BoundaryCondition(object):
    def __init__(self, grid):
        self.grid = grid

class Periodic(BoundaryCondition):
    def __call__(self, u, t):
        n_ghost = self.grid.n_ghost

        u[:,:n_ghost,...] = u[:,-2*n_ghost:-n_ghost,...]
        u[:,-n_ghost:,...] = u[:,n_ghost:2*n_ghost,...]

        if self.grid.n_dims == 2:
            u[:,:,:n_ghost] = u[:,:,-2*n_ghost:-n_ghost]
            u[:,:,-n_ghost:] = u[:,:,n_ghost:2*n_ghost]


class Outflow(BoundaryCondition):
    def __call__(self, u, t):
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
        super().__init__(grid)
        self.equilibrium = equilibrium

    def __call__(self, u, t):
        n_ghost = self.grid.n_ghost
        self.set_layer(u, n_ghost, slice(0, n_ghost))
        self.set_layer(u, -n_ghost-1, slice(-n_ghost, None))

    def set_layer(self, u, i_inner, i_outer):
        u_ref = u[:,i_inner,...]
        x_ref = self.grid.cell_centers[i_inner,...,0]
        x = self.grid.cell_centers[i_outer,...,0]

        u[:,i_outer,...] = self.equilibrium.reconstruct(u_ref, x_ref, x)


class PeriodicVelocity(BoundaryCondition):
    def __init__(self, grid, equilibrium):
        super().__init__(grid)
        self.hydrostatic_bc = HydrostaticOutflow(grid, equilibrium)
        self.amplitude = 1e-5
        self.frequency = 8*np.pi

        self.u0 = None

    def __call__(self, u, t):
        model = self.hydrostatic_bc.equilibrium.model
        n_ghost = self.grid.n_ghost

        if self.u0 is None:
            self.u0 = np.copy(u[:,:n_ghost])


        # E_kin = model.kinetic_energy(u[:,n_ghost])
        # print(E_kin)
        # mvx = u[1,n_ghost]
        # u[1,n_ghost] = 0.0
        # u[3,n_ghost] = u[3,n_ghost] - E_kin
        self.hydrostatic_bc(u, t)
        u[0,:n_ghost] = self.u0[0,...]
        # u[3,n_ghost,...] += E_kin
        # u[1,n_ghost] = mvx

        A, omega = self.amplitude, self.frequency
        u[1,:n_ghost,...] = u[0,:n_ghost,...] * A*np.sin(omega*t)
        u[3,:n_ghost,...] += 0.5*u[0,:n_ghost,...]*(A*np.sin(omega*t))**2
