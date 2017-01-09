import numpy as np

from advection import Advection
from rusanov import Rusanov
from grid import Grid
from boundary_conditions import Periodic
from finite_volume_fluxes import FiniteVolumeFluxesO1
from runge_kutta import ForwardEuler
from time_loop import TimeLoop
from time_keeper import FixedDuration, PlotNever

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
    shape = pde.grid.cell_centers.shape[:2] + (1,)

    visualize = lambda u : None
    plotting_steps = PlotNever()
    single_step = ForwardEuler(pde.bc, pde.fvm, shape)
    simulation = TimeLoop(single_step, visualize, plotting_steps)

    u0 = (np.cos(2*np.pi*pde.grid.cell_centers[:,:,0])
          * np.sin(2*np.pi*pde.grid.cell_centers[:,:,1])).reshape(shape)

    time_keeper = FixedDuration(T = 0.3)
    uT = simulation(u0, time_keeper);

    assert np.all(np.isfinite(uT))


