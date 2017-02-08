#!/usr/bin/env python3
# encoding: utf-8

import numpy as np

from equilibrium import IsothermalEquilibrium
from euler_experiment import EulerExperiment
from boundary_conditions import Outflow, IsothermalOutflow
from visualize import EquilibriumGraphs
from weno import OptimalWENO, EquilibriumStencil
from source_terms import BalancedSourceTerm
from math_tools import gaussian, l1_error, convergence_rate
from time_keeper import FixedSteps, PlotNever

class GaussianBumpIC(object):
    def __init__(self, model):
        self.model = model
        self.amplitude = 1.0e-5
        self.sigma = 0.05
        self.x_ref = 0.0
        self.x_mid = 0.5
        self.rho_ref = 1.0
        self.p_ref = 1.0
        E_int_ref = model.internal_energy(p=self.p_ref)
        self.u_ref = np.array([self.rho_ref, 0.0, 0.0, E_int_ref])

    def __call__(self, grid):
        x = grid.cell_centers[:,0]

        u0 = self.back_ground(grid)
        alpha = 1.0 + self.amplitude*gaussian(x - self.x_mid, self.sigma)
        # u0[0,...] = u0[0,...]/alpha
        u0[3,...] = u0[3,...]*alpha

        return u0

    def back_ground(self, grid):
        x = grid.cell_centers[:,0]
        equilibrium = IsothermalEquilibrium(model=self.model, grid=grid)
        u_bar_ref = equilibrium.cell_averages(self.u_ref)

        return equilibrium.reconstruct(u_bar_ref, self.x_ref, x)


class GaussianBump(EulerExperiment):
    @property
    def final_time(self):
        return 0.2

    @property
    def n_cells(self):
        return getattr(self, "_n_cells", 200)

    @n_cells.setter
    def n_cells(self, new_value):
        self._n_cells = new_value

    @property
    def initial_condition(self):
        return GaussianBumpIC(self.model)

    @property
    def order(self):
        return 5

    @property
    def gravity(self):
        return 1.0

    @property
    def boundary_condition(self):
        return IsothermalOutflow(self.grid, self.model)

    @property
    def weno(self):
        return OptimalWENO(EquilibriumStencil(self.grid, self.equilibrium, self.model))

    @property
    def source(self):
        return BalancedSourceTerm(self.grid, self.model, self.equilibrium)

    @property
    def equilibrium(self):
        return IsothermalEquilibrium(self.grid, self.model)

    @property
    def visualize(self):
        back_ground = self.initial_condition.back_ground(self.grid)
        return EquilibriumGraphs(self.grid, self.output_filename, self.model, back_ground)

    @property
    def output_filename(self):
        return "img/gaussian_bump"

    @property
    def steps_per_frame(self):
        return 2


if __name__ == '__main__':
    gaussian_bump = GaussianBump()
    gaussian_bump()
