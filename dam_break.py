#!/usr/bin/env python3
# encoding: utf-8

# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

import numpy as np

from morinth.boundary_conditions import Outflow
from morinth.euler_experiment import ShallowWaterExperiment

class DamBreakIC(object):
    def __call__(self, grid):
        x = grid.cell_centers

        x_crit = grid.lower_boundary[0] + 0.3*grid.extent[0]
        print(x_crit)

        u0 = np.zeros((2,) + grid.cell_centers.shape[:1])
        u0[0,...] = np.where(x[...,0] < x_crit, 5.0, 1.0)

        return u0


class DamBreak(ShallowWaterExperiment):
    @property
    def n_cells(self):
        return 200

    @property
    def order(self):
        return 5

    @property
    def output_filename(self):
        return "img/dam_break"

    @property
    def steps_per_frame(self):
        return 5

    @property
    def final_time(self):
        return 0.1

    @property
    def initial_condition(self):
        return DamBreakIC()

    @property
    def boundary_condition(self):
        return Outflow(self.grid)


if __name__ == '__main__':
    dam_break = DamBreak()
    dam_break()
