import numpy as np

from burgers import Burgers
from rusanov import Rusanov
from grid import Grid
from boundary_conditions import Periodic
from finite_volume_fluxes import FiniteVolumeFluxesO1
from time_integration import ForwardEuler


def test_forward_euler_init():
    model = Burgers()
    flux = Rusanov(model)
    grid = Grid([0.0, 1.0], 100, 1)
    fvm = FiniteVolumeFluxesO1(grid, flux)
    bc = Periodic(grid)

    forward_euler = ForwardEuler(grid, bc, fvm)

    assert forward_euler.bc is bc
    assert forward_euler.rate_of_change is fvm
