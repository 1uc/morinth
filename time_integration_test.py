import numpy as np

from burgers import Burgers
from rusanov import Rusanov
from grid import Grid
from boundary_conditions import Periodic
from finite_volume_fluxes import FiniteVolumeFluxesO1
from time_integration import BackwardEuler, BDF2, DIRKa23, DIRKa34
from runge_kutta import ForwardEuler, SSP2, SSP3
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
    bc = lambda x: None
    mock_roc = MockROC()
    mask = np.array([True])

    solvers = [ ForwardEuler(bc, mock_roc),
                BackwardEuler(bc, mock_roc, mask, 3.0),
                BDF2(bc, mock_roc, mask, fixed_dt = 0.05),
                SSP2(bc, mock_roc),
                SSP3(bc, mock_roc),
                DIRKa23(bc, mock_roc, mask, 3.0),
                DIRKa34(bc, mock_roc, mask, 3.0)]

    tolerances = [0.01, 0.05, 0.008, 0.01, 0.004, 0.0082, 0.0017]

    T = 1.0

    for single_step, tol in zip(solvers, tolerances):
        plotting_steps = PlotNever()
        time_loop = TimeLoop(single_step, lambda x: None, plotting_steps)

        u0 = np.array([1.0]).reshape((1, 1, 1))
        uT = time_loop(u0, FixedDuration(T))

        assert np.all(np.abs(uT - u0*np.exp(T)) < tol), str(single_step)

if __name__ == '__main__':
    test_mock_ode()
