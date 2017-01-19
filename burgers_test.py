import numpy as np
from burgers import Burgers
from rusanov import Rusanov
from grid import Grid
from boundary_conditions import Periodic
from finite_volume_fluxes import FiniteVolumeFluxes
from time_integration import BackwardEuler
from runge_kutta import ForwardEuler
from time_loop import TimeLoop
from visualize import SimpleGraph
from time_keeper import FixedDuration, PlotAtFixedInterval, PlotEveryNthStep

import pytest

def test_burgers_eigenvalue():
    model = Burgers()
    u = np.random.random((1, 100))

    assert np.all(model.max_eigenvalue(u) >= 0.0)
    assert np.all(model.max_eigenvalue(u) == np.abs(u))

class PDE(object):
    def __init__(self):
        self.grid = Grid([0.0, 1.0], 100, 1)
        self.model = Burgers()
        self.flux = Rusanov(self.model)
        self.fvm = FiniteVolumeFluxes(self.grid, self.flux)
        self.bc = Periodic(self.grid)


def test_burgers():
    pde = PDE()

    forward_euler = ForwardEuler(pde.bc, pde.fvm)
    backward_euler = BackwardEuler(pde.bc, pde.fvm, pde.grid.boundary_mask, cfl_number=3.0)
    solvers = [backward_euler]
    labels = ["backward_euler"]

    T = 0.3
    u0 = np.cos(2*np.pi*pde.grid.cell_centers.reshape((1, -1, 1)))
    for single_step, label in zip(solvers, labels):
        visualize = lambda x: None
        plotting_steps = PlotEveryNthStep(1)
        simulation = TimeLoop(single_step, visualize, plotting_steps)
        uT = simulation(u0, FixedDuration(T));

        assert np.all(np.isfinite(uT))
