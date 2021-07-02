#!/usr/bin/env python3
# encoding: utf-8

# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

import numpy as np

from morinth.boundary_conditions import Periodic, ZeroBurgersBoundary
from morinth.visualize import DumpToDiskWithTime
from morinth.time_keeper import PlotAtFixedInterval
from morinth.euler_experiment import BurgersExperiment


class BurgersSineWave(BurgersExperiment):
    @property
    def domain(self):
        return [-1.0, 1.0]

    @property
    def n_cells(self):
        return 1000

    @property
    def final_time(self):
        return 0.2

    @property
    def order(self):
        return 5

    @property
    def initial_condition(self):
        return lambda grid: np.sin(2.0 * np.pi * grid.cell_centers).reshape((1, -1))

    @property
    def boundary_condition(self):
        return ZeroBurgersBoundary(self.grid)

    @property
    def output_filename(self):
        return "img/burgers_sine_wave"

    @property
    def visualize(self):
        return DumpToDiskWithTime(self.grid, self.output_filename, self.time_keeper)

    @property
    def plotting_steps(self):
        return PlotAtFixedInterval(dt_vis=0.05)


if __name__ == "__main__":
    burgers_sine_wave = BurgersSineWave()
    burgers_sine_wave()
