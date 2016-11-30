import numpy as np

from burgers import Burgers
from rusanov import Rusanov
from grid import Grid
from boundary_conditions import Periodic
from finite_volume_fluxes import FiniteVolumeFluxesO1
from time_integration import ForwardEuler, BackwardEuler, BDF2
from time_loop import TimeLoop
from time_keeper import FixedDuration, PlotNever


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
    backward_euler = BackwardEuler(lambda x: None, mock_roc, np.array([True]))
    bdf2 = BDF2(lambda x: None, mock_roc, np.array([True]), fixed_dt = 0.05)
    solvers = [forward_euler, backward_euler, bdf2]
    tolerances = [0.01, 0.05, 0.008]

    T = 1.0

    for single_step, tol in zip(solvers, tolerances):
        plotting_steps = PlotNever()
        time_loop = TimeLoop(single_step, lambda x: None, plotting_steps)

        u0 = np.array([1.0]).reshape((1, 1, 1))
        uT = time_loop(u0, FixedDuration(T))

        assert np.all(np.abs(uT - u0*np.exp(T)) < tol)

if __name__ == '__main__':
    test_mock_ode()
