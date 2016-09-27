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

def test_burgers_eigenvalue():
    model = Burgers()
    u = np.random.random((100, 1))

    assert np.all(model.max_eigenvalue(u) >= 0.0)

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
    return ForwardEuler(bc, fvm)

def test_burgers(grid, single_step):
    visualize = lambda u : None
    simulation = TimeLoop(single_step, visualize)

    u0 = np.cos(2*np.pi*grid.cell_centers.reshape((-1, 1, 1)))

    T = 0.3
    uT = simulation(u0, T);

    assert np.all(np.isfinite(uT))
