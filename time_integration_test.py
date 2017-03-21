import time
import numpy as np

from burgers import Burgers
from rusanov import Rusanov
from grid import Grid
from boundary_conditions import Periodic
from time_integration import BackwardEuler, BDF2, DIRKa23, DIRKa34
from runge_kutta import ForwardEuler, SSP2, SSP3, Fehlberg
from time_loop import TimeLoop
from time_keeper import FixedDuration, PlotNever, FixedSteps
from math_tools import convergence_rate


class MockROC(object):
    def __init__(self, dt):
        self._dt = dt

    def __call__(self, u, t):
        return u

    def pick_time_step(self, u):
        return self._dt

def with_mask(Solver, mask, cfl_number):
    return lambda bc, mock_roc: Solver(bc, mock_roc, mask, cfl_number)

def test_mock_ode():
    bc = lambda x: None
    plotting_steps = PlotNever()
    mask = np.array([True])

    solvers = [ ForwardEuler,
                with_mask(BackwardEuler, mask, 3.0),
                SSP2,
                SSP3,
                Fehlberg
                ]

    rates = [ 1.0, 1.0, 2.0, 3.0, 5.0 ]

    assert len(solvers) == len(rates)

    T = 1.0
    all_resolutions = [10, 20, 40]
    error = np.empty(len(all_resolutions))

    for Solver, expected_rate in zip(solvers, rates):
        for k, res in enumerate(all_resolutions):
            mock_roc = MockROC(T/res)
            single_step = Solver(bc, mock_roc)
            time_loop = TimeLoop(single_step, lambda x: None, plotting_steps)

            u0 = np.array([1.0]).reshape((1, 1, 1))
            uT = time_loop(u0, FixedDuration(T))

            u_ref = u0*np.exp(T)
            # u_ref = u0 + np.array([T])
            error[k] = np.abs(uT - u_ref)

        observed_rate = np.abs(convergence_rate(error, np.array(all_resolutions)))
        assert np.all(observed_rate - expected_rate > -0.1), str(single_step)

if __name__ == '__main__':
    bc = lambda x: None
    plotting_steps = PlotNever()

    solvers = [ ForwardEuler,
                SSP2,
                SSP3,
                Fehlberg
              ]

    n_steps, dt = 100000, 1e-6

    for Solver in solvers:
        single_step = Solver(bc, MockROC(dt))
        time_loop = TimeLoop(single_step, lambda x: None, plotting_steps)

        u0 = np.random.random((1000, 1, 1))

        t0 = time.perf_counter()
        uT = time_loop(u0, FixedSteps(n_steps))
        t1 = time.perf_counter()

        print("{:s} : {:.3f} s".format(Solver.__name__, t1 - t0))
