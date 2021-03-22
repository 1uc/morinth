import numpy as np

from morinth.advection import Advection
from morinth.burgers import Burgers, VariableBurgers
from morinth.euler import Euler
from morinth.shallow_water import ShallowWater

from morinth.hllc import HLLC
from morinth.rusanov import Rusanov

from morinth.weno import StableWENO, ENO, OptimalWENO
from morinth.runge_kutta import ForwardEuler, SSP3, Fehlberg

from morinth.grid import Grid
from morinth.time_keeper import PlotEveryNthStep, FixedDuration
from morinth.boundary_conditions import Periodic
from morinth.visualize import SimpleGraph, EulerGraphs, EulerColormaps

from morinth.finite_volume_fluxes import FVMRateOfChange, FirstOrderReconstruction
from morinth.time_loop import TimeLoop
from morinth.progress_bar import ProgressBar


class NumericalExperiment(object):
    """Base for all numerical experitments."""

    def __call__(self):
        u0 = self.initial_condition(self.grid)
        return self.simulation(u0, self.time_keeper)

    @property
    def n_ghost(self):
        if self.order == 1:
            return 1
        elif self.order == 3 or self.order == 5:
            return 3
        else:
            raise Exception("Invalid order [{:s}].".format(self.order))

    @property
    def grid(self):
        if hasattr(self, "_grid"):
            return self._grid
        else:
            self._grid = Grid(self.domain, self.n_cells, self.n_ghost)
            return self._grid

    @property
    def domain(self):
        return [0.0, 1.0]

    @property
    def fvm(self):
        return FVMRateOfChange(self.grid, self.flux, self.reconstruction, self.source)

    @property
    def source(self):
        return None

    @property
    def single_step(self):
        if hasattr(self, "_single_step"):
            return self._single_step
        else:
            if self.order == 1:
                self._single_step = ForwardEuler(self.boundary_condition, self.fvm)
            elif self.order == 3:
                self._single_step = SSP3(self.boundary_condition, self.fvm)
            elif self.order == 5:
                self._single_step = Fehlberg(self.boundary_condition, self.fvm)
            else:
                raise Exception("Invalid order [{:s}].".format(self.order))

            return self._single_step

    @property
    def time_keeper(self):
        if hasattr(self, "_time_keeper"):
            return self._time_keeper
        else:
            self._time_keeper = self._make_time_keeper()
            return self._time_keeper

    def _make_time_keeper(self):
        return FixedDuration(self.final_time, self.needs_baby_steps)

    @property
    def needs_baby_steps(self):
        return False

    @property
    def simulation(self):
        if hasattr(self, "_simulation"):
            return self._simulation
        else:
            self._simulation = TimeLoop(
                self.single_step, self.visualize, self.plotting_steps, self.progress_bar
            )
            return self._simulation

    @property
    def plotting_steps(self):
        return PlotEveryNthStep(steps_per_frame=self.steps_per_frame)

    @property
    def steps_per_frame(self):
        return 30

    @property
    def flux(self):
        return Rusanov(self.model)

    @property
    def reconstruction(self):
        if self.order == 1:
            return FirstOrderReconstruction()
        elif self.order == 3:
            return self.eno
        elif self.order == 5:
            return self.weno
        else:
            raise Exception("Invalid order [{:s}].".format(self.order))

    @property
    def progress_bar(self):
        return ProgressBar(10)

    @property
    def eno(self):
        return ENO()

    @property
    def weno(self):
        return OptimalWENO()


class AdvectionExperiment(NumericalExperiment):
    @property
    def model(self):
        return Advection(self.velocity)

    @property
    def boundary_condition(self):
        return Periodic(self.grid)


class BurgersExperiment(NumericalExperiment):
    @property
    def model(self):
        return Burgers()

    @property
    def visualize(self):
        return SimpleGraph(self.grid, self.output_filename)


class VariableBurgersExperiment(NumericalExperiment):
    @property
    def a(self):
        return self.model.a

    @a.setter
    def a(self, a):
        self.model.a = a

    @property
    def model(self):
        if not hasattr(self, "_model"):
            self._model = VariableBurgers(a=None)

        return self._model

    @property
    def visualize(self):
        return SimpleGraph(self.grid, self.output_filename)


class ShallowWaterExperiment(NumericalExperiment):
    """Base class for all shallow-water numerical experiments."""

    @property
    def model(self):
        return ShallowWater(gravity=self.gravity)

    @property
    def gravity(self):
        return 10.0

    @property
    def visualize(self):
        return SimpleGraph(self.grid, self.output_filename)


class EulerExperiment(NumericalExperiment):
    """Fixture for numerical experiments with Euler equations."""

    @property
    def model(self):
        return Euler(
            gravity=self.gravity,
            gamma=self.gamma,
            specific_gas_constant=self.specific_gas_constant,
        )

    @property
    def gravity(self):
        return 0.0

    @property
    def gamma(self):
        return 1.4

    @property
    def specific_gas_constant(self):
        return 1.0

    @property
    def visualize(self):
        return EulerGraphs(self.grid, self.output_filename, self.model)

    @property
    def flux(self):
        return HLLC(self.model)


class EulerExperiment2D(EulerExperiment):
    @property
    def domain(self):
        return [[0.0, 1.0], [0.0, 1.0]]

    @property
    def resolution(self):
        return (self.n_cells, self.n_cells)

    @property
    def visualize(self):
        return EulerColormaps(self.grid, self.output_filename, self.model)

    @property
    def grid(self):
        if hasattr(self, "_grid"):
            return self._grid
        else:
            self._grid = Grid(self.domain, self.resolution, self.n_ghost)
            return self._grid
