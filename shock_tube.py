#!/usr/bin/env python
# encoding: utf-8

import numpy as np

from grid import Grid
from visualize import EulerGraphs
from time_loop import TimeLoop
from time_keeper import PlotEveryNthStep, FixedDuration, PlotNever
from boundary_conditions import Outflow

from euler_experiment import EulerExperiment, scheme_o1

class ShockTubeIC:
    """Toro's shock-tube."""

    def __init__(self, model):
        self.model = model

    def __call__(self, grid):
        u0 = np.empty((4, grid.cell_centers.shape[0]))

        x_crit = grid.edges[0] + 0.4*(grid.edges[-1] - grid.edges[0])
        is_inside = grid.cell_centers[...,0] < x_crit

        u0[0, ...] = np.where(is_inside, 1.0, 0.125)
        u0[1, ...] = np.where(is_inside, 0.75, 0.0)
        u0[2, ...] = 0.0
        u0[3, ...] = np.where(is_inside, 1.0, 0.1)

        return self.model.conserved_variables(u0)


class ShockTube(EulerExperiment):

    def __call__(self):
        ic = ShockTubeIC(self.model)
        u0 = ic(self.grid)

        _, _, single_step = scheme_o1(self.model, self.grid, self.boundary_conditions)
        time_keeper = FixedDuration(0.20)
        simulation = TimeLoop(single_step, self.visualize, self.plotting_steps, self.progress_bar)
        simulation(u0, time_keeper)

    def set_up_grid(self):
        self.grid = Grid([0.0, 1.0], 800, 1)

    def set_up_visualization(self):
        self.visualize = EulerGraphs(self.grid, "img/shock_tube", self.model)
        self.plotting_steps = PlotEveryNthStep(steps_per_frame = 30)

    def set_up_boundary_condition(self):
        self.boundary_conditions = Outflow(self.grid)

if __name__ == '__main__':
    shock_tube = ShockTube()
    shock_tube()
