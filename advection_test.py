import numpy as np

import matplotlib.pylab as plt

from advection import Advection
from rusanov import Rusanov
from grid import Grid
from boundary_conditions import Periodic
from finite_volume_fluxes import FVMRateOfChange
from runge_kutta import ForwardEuler, Fehlberg
from time_loop import TimeLoop
from time_keeper import FixedDuration, PlotNever
from quadrature import GaussLegendre
from math_tools import l1_error, convergence_rate
from euler_experiment import AdvectionExperiment
from progress_bar import SilentProgressBar
from weno import OptimalWENO

import pytest

class PDE(object):
    def __init__(self):
        self.grid = Grid([[0.0, 1.0], [-1.0, -0.5]], [100, 20], 1)
        self.model = Advection(np.array([1.0, 2.0]))
        self.flux = Rusanov(self.model)
        self.fvm = FVMRateOfChange(self.grid, self.flux, None, None)
        self.bc = Periodic(self.grid)

def test_advection():
    pde = PDE()

    visualize = lambda u : None
    plotting_steps = PlotNever()
    single_step = ForwardEuler(pde.bc, pde.fvm)
    simulation = TimeLoop(single_step, visualize, plotting_steps)

    shape = (1,) + pde.grid.cell_centers.shape[:2]
    u0 = (np.cos(2*np.pi*pde.grid.cell_centers[:,:,0])
          * np.sin(2*np.pi*pde.grid.cell_centers[:,:,1])).reshape(shape)

    time_keeper = FixedDuration(T = 0.3)
    uT = simulation(u0, time_keeper);

    assert np.all(np.isfinite(uT))

def smooth_pattern(x):
    return np.sin(2.0*np.pi*x - 0.1)

class SineAdvection(AdvectionExperiment):
    def __init__(self, n_cells, order):
        self._order = order
        self._n_cells = n_cells + self.n_ghost
        self.cell_average = GaussLegendre(5)

    @property
    def final_time(self):
        return 0.05

    @property
    def progress_bar(self):
        return SilentProgressBar()

    @property
    def plotting_steps(self):
        return PlotNever()

    @property
    def order(self):
        return self._order

    @property
    def n_cells(self):
        return self._n_cells

    @property
    def initial_condition(self):
        return lambda grid: self.cell_average(grid.edges, smooth_pattern).reshape((1, -1))

    @property
    def velocity(self):
        return np.array([2.34])

    @property
    def reference_solution(self):
        t, v = self.final_time, self.velocity
        return self.cell_average(self.grid.edges, lambda x: smooth_pattern(x - t*v)).reshape((1, -1))
        # return smooth_pattern(self.grid.cell_centers - t*v).reshape((1, -1))

    @property
    def visualize(self):
        return None

    @property
    def single_step(self):
        _single_step = Fehlberg(self.boundary_condition, self.fvm)
        self._single_step = getattr(self, "_single_step", _single_step)
        return self._single_step


def test_convergence_rate():
    all_resolutions = np.array([10, 20, 40, 80, 160, 320, 640])

    weno = OptimalWENO()
    err = np.empty(all_resolutions.size)

    for k, resolution in enumerate(np.nditer(all_resolutions)):
        simulation = SineAdvection(resolution, 5)
        grid = simulation.grid

        uT = simulation()
        u_ref = simulation.reference_solution

        # plt.clf()
        # plt.plot(grid.cell_centers[:,0], uT[0,...])
        # plt.hold(True)
        # plt.plot(grid.cell_centers[:,0], u_ref[0,...])
        # plt.show()

        err[k] = l1_error(uT[:,3:-3], u_ref[:,3:-3])

    rate = convergence_rate(err, all_resolutions-6)
    assert np.abs(np.max(np.abs(rate)) - 5.0) < 0.1

    # print("")
    # print(err)
    # print(convergence_rate(err, np.array(all_resolutions)-6))

