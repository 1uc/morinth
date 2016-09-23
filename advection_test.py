import numpy as np

from advection import Advection
from rusanov import Rusanov
from grid import Grid
from boundary_conditions import Periodic
from finite_volume_fluxes import FiniteVolumeFluxesO1
from time_integration import ForwardEuler
from time_loop import TimeLoop
from visualize import SimpleColormap

import pytest

@pytest.fixture
def grid():
    return Grid([[0.0, 1.0], [-1.0, -0.5]], [100, 20], 1)

@pytest.fixture
def model():
    return Advection(np.array([1.0, 2.0]))

@pytest.fixture
def single_step(grid, model):
    flux = Rusanov(model)
    fvm = FiniteVolumeFluxesO1(grid, flux)
    bc = Periodic(grid)
    return ForwardEuler(grid, bc, fvm)

def test_advection(grid, single_step):
    visualize = SimpleColormap(grid, "advection_test")
    simulation = TimeLoop(single_step, visualize)

    shape = grid.cell_centers.shape[:2] + (1,)
    u0 = (np.cos(2*np.pi*grid.cell_centers[:,:,0])
          * np.sin(2*np.pi*grid.cell_centers[:,:,1])).reshape(shape)

    T = 0.3
    uT = simulation(u0, T);
