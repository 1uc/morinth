#!/usr/bin/env python3
# encoding: utf-8

import numpy as np

from equilibrium import IsothermalEquilibrium, IsothermalRC
from euler_experiment import EulerExperiment
from boundary_conditions import Outflow, IsothermalOutflow
from visualize import EquilibriumGraphs
from weno import OptimalWENO, EquilibriumStencil
from source_terms import BalancedSourceTerm
from math_tools import gaussian

class GaussianBumpIC(object):
    def __init__(self, model):
        self.model = model
        self.amplitude = 0.0
        self.sigma = 0.05
        self.p_ref = 1.0
        self.T_ref = 1.0
        self.x_ref = 0.0
        self.x_mid = 0.5

    def __call__(self, grid):
        x = grid.cell_centers[:,0]

        w0 = self.back_ground(grid)
        w0[3,...] = w0[3,...]*(1.0 + self.amplitude*gaussian(x - self.x_mid, self.sigma))

        return self.model.conserved_variables(w0)

    def back_ground(self, grid):
        x = grid.cell_centers[:,0]
        equilibrium = IsothermalEquilibrium(self.model, self.p_ref, self.T_ref, self.x_ref)

        w0 = np.zeros((4, grid.n_cells[0]))
        w0[0,...], w0[3,...] = equilibrium(x)
        return w0


class GaussianBump(EulerExperiment):
    @property
    def final_time(self):
        return 0.1

    @property
    def n_cells(self):
        return 200

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
        # return OptimalWENO()

    @property
    def source(self):
        return BalancedSourceTerm(self.grid, self.model, self.equilibrium)

    @property
    def equilibrium(self):
        return IsothermalRC(self.grid, self.model)

    @property
    def visualize(self):
        back_ground = self.initial_condition.back_ground(self.grid)
        back_ground = self.model.conserved_variables(back_ground)
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
