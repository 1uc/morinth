import numpy as np

from euler import Euler
from hllc import HLLC
from rusanov import Rusanov
from weno import ENO, OptimalWENO
from runge_kutta import ForwardEuler, SSP3
from source_terms import CenteredSourceTerm

class FVMRateOfChange:
    def __init__(self, grid, flux, reconstruction, source=None):
        self.finite_volume_fluxes = FiniteVolumeFluxes(grid, flux, reconstruction)
        if source is None:
            self.source = CenteredSourceTerm(flux.model)
        else:
            self.source = source

    def __call__(self, u, t):
        dudt = self.finite_volume_fluxes(u, t)
        dudt += self.source(u, t)

        return dudt

    def pick_time_step(self, u):
        return self.finite_volume_fluxes.pick_time_step(u)


class FiniteVolumeFluxes(object):
    """Rate of change due to the finite volume fluxes."""

    def __init__(self, grid, flux, reconstruction = None):
        self.grid = grid
        self.flux = flux
        self.model = flux.model

        if reconstruction is None:
            reconstruction = FirstOrderReconstruction()

        self.reconstruction = reconstruction

    def __call__(self, u, t):
        """Compute the rate of change 'dudt' due to FVM."""
        n_ghost = self.grid.n_ghost
        I = slice(n_ghost, -n_ghost)

        dudt = np.zeros_like(u)
        dudt[:,I,...] = self.x_flux(u, t)

        if self.grid.n_dims == 2:
            dudt[:,:,I] += self.y_flux(u, t)

        return dudt

    def x_flux(self, u, t):
        u_left, u_right = self.reconstruction(u, axis=0)
        flux = self.flux(u_left, u_right, axis=0)
        return -1.0/self.grid.dx * (flux[:,1:,...] - flux[:,:-1,...])

    def y_flux(self, u, t):
        u_left, u_right = self.reconstruction(u, axis=1)
        flux = self.flux(u_left, u_right, axis=1)

        return -1.0/self.grid.dy * (flux[:,:,1:] - flux[:,:,:-1])

    def pick_time_step(self, u):
        return self.grid.dx/np.max(self.model.max_eigenvalue(u));


class FirstOrderReconstruction():
    def __call__(self, u, axis):
        return self.left(u, axis), self.right(u, axis)

    def left(self, u, axis):
        if axis == 0:
            return u[:,:-1,...]
        elif axis == 1:
            return u[:,:,:-1]

    def right(self, u, axis):
        if axis == 0:
            return u[:,1:,...]
        elif axis == 1:
            return u[:,:,1:]
