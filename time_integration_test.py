import numpy as np

from burgers import Burgers
from rusanov import Rusanov
from grid import Grid
from boundary_conditions import Periodic
from finite_volume_fluxes import FiniteVolumeFluxesO1
from time_integration import ForwardEuler
from time_loop import TimeLoop


def test_forward_euler_init():
    model = Burgers()
    flux = Rusanov(model)
    grid = Grid([0.0, 1.0], 100, 1)
    fvm = FiniteVolumeFluxesO1(grid, flux)
    bc = Periodic(grid)

    forward_euler = ForwardEuler(bc, fvm)

    assert forward_euler.bc is bc
    assert forward_euler.rate_of_change is fvm

class MockROC(object):
    def __call__(self, u, t):
        return u

    def pick_time_step(self, u):
        return 0.01

def test_mock_ode():
    mock_roc = MockROC()
    forward_euler = ForwardEuler(lambda x: None, mock_roc)
    solvers = [forward_euler]
    tolerances = [0.01]

    T = 1.0

    for single_step, tol in zip(solvers, tolerances):
        time_loop = TimeLoop(single_step, lambda x: None)

        u0 = np.array([1.0]).reshape((1, 1, 1))
        uT = time_loop(u0, T)

        assert np.all(np.abs(uT - u0*np.exp(T)) < tol)
