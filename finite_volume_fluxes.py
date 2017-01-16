import numpy as np

class FiniteVolumeFluxesO1(object):
    """Rate of change due to the finite volume fluxes."""

    def __init__(self, grid, flux):
        self.grid = grid
        self.flux = flux
        self.model = flux.model

    def __call__(self, u, t):
        """Compute the rate of change 'dudt' due to FVM."""
        dudt = np.zeros_like(u)
        dudt[:,1:-1,:] = self.x_flux(u, t)

        if self.grid.n_dims == 2:
            dudt[:,:,1:-1] += self.y_flux(u, t)

        return dudt

    def x_flux(self, u, t):
        return -1.0/self.grid.dx * (self.flux(u[:,1:-1,:], u[:,2:,:], 0)
                                    - self.flux(u[:,:-2,:], u[:,1:-1,:], 0))

    def y_flux(self, u, t):
        return -1.0/self.grid.dy * (self.flux(u[:,:,1:-1], u[:,:,2:], 1)
                                    - self.flux(u[:,:,:-2], u[:,:,1:-1], 1))


    def pick_time_step(self, u):
        return self.grid.dx/np.max(self.model.max_eigenvalue(u));

