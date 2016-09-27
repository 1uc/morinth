import numpy as np

from advection import Advection
from rusanov import Rusanov
from grid import Grid
from boundary_conditions import Periodic
from finite_volume_fluxes import FiniteVolumeFluxesO1
from time_integration import ForwardEuler
from time_loop import TimeLoop

import pytest

class PDE(object):
    def __init__(self):
        self.grid = Grid([[0.0, 1.0], [-1.0, -0.5]], [100, 20], 1)
        self.model = Advection(np.array([1.0, 2.0]))
        self.flux = Rusanov(self.model)
        self.fvm = FiniteVolumeFluxesO1(self.grid, self.flux)
        self.bc = Periodic(self.grid)

def test_advection():
    pde = PDE()

    visualize = lambda u : None
    single_step = ForwardEuler(pde.bc, pde.fvm)
    simulation = TimeLoop(single_step, visualize)

    shape = pde.grid.cell_centers.shape[:2] + (1,)
    u0 = (np.cos(2*np.pi*pde.grid.cell_centers[:,:,0])
          * np.sin(2*np.pi*pde.grid.cell_centers[:,:,1])).reshape(shape)

    T = 0.3
    uT = simulation(u0, T);

    assert np.all(np.isfinite(uT))


