import numpy as np

from euler import Euler
from hllc import HLLC
from rusanov import Rusanov
from weno import ENO, OptimalWENO
from runge_kutta import ForwardEuler, SSP3
from source_terms import CenteredSourceTerm

from coding_tools import with_default

class FVMRateOfChange:
    def __init__(self, grid, flux, reconstruction=None, source=None):
        self.grid = grid
        self.flux = flux
        self.model = flux.model
        self.reconstruction = with_default(reconstruction, FirstOrderReconstruction())
        self.source = with_default(source, CenteredSourceTerm(flux.model))

    def __call__(self, u, t):
        n_ghost = self.grid.n_ghost
        I = slice(n_ghost, -n_ghost)

        dudt = np.zeros_like(u)

        # NOTATION:
        # The `_plus` trace in from the left, `_minus` is from the right of the
        # interface.
        # The `_left` trace is left of the cell, and `_right` is right of the
        # cell.
        u_plus, u_minus = self.reconstruction(u, axis=0)
        dudt[:,I,...] = self.x_flux(u, u_plus, u_minus)

        if self.source.needs_edge_source:
            dudt[:,I,...] += self.x_source(u, u_plus, u_minus)


        if self.grid.n_dims == 2:
            u_plus, u_minus = self.reconstruction(u, axis=1)
            dudt[:,:,I] += self.y_flux(u, u_plus, u_minus)

            if self.source.needs_edge_source:
                dudt[:,:,I] += self.y_source(u, u_plus, u_minus)

        if self.source.needs_volume_source:
            dudt += self.source.volume_source(u, u_plus, u_minus)

        return dudt

    def x_flux(self, u, u_plus, u_minus):
        flux = self.flux(u_plus, u_minus, axis=0)
        return -1.0/self.grid.dx * (flux[:,1:,...] - flux[:,:-1,...])

    def x_source(self, u, u_plus, u_minus):
        return self.source.edge_source(u, u_minus[:,:-1,...], u_plus[:,1:,...], axis=0)

    def y_flux(self, u, u_plus, u_minus):
        flux = self.flux(u_plus, u_minus, axis=1)
        return -1.0/self.grid.dy * (flux[:,:,1:] - flux[:,:,:-1])

    def y_source(self, u, u_plus, u_minus):
        return self.source.edge_source(u, u_plus, u_minus, axis=1)

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
