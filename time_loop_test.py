import numpy as np

from burgers import Burgers
from rusanov import Rusanov
from grid import Grid
from boundary_conditions import Periodic
from finite_volume_fluxes import FiniteVolumeFluxesO1
from time_integration import ForwardEuler
from time_loop import TimeLoop
from visualize import SimpleGraph

import pytest

@pytest.fixture
def grid():
    return Grid([0.0, 1.0], 100, 1)

@pytest.fixture
def model():
    return Burgers()

@pytest.fixture
def single_step(grid, model):
    flux = Rusanov(model)
    fvm = FiniteVolumeFluxesO1(grid, flux)
    bc = Periodic(grid)
    return ForwardEuler(grid, bc, fvm)

def test_sine_wave(grid, single_step):
    visualize = SimpleGraph(grid, "burgers_test")
    simulation = TimeLoop(single_step, visualize)

    u0 = np.cos(2*np.pi*grid.cell_centers.reshape((-1, 1, 1)))

    T = 0.3
    uT = simulation(u0, T);
