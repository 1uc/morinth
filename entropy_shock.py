#!/usr/bin/env python3
# encoding: utf-8

import numpy as np

from morinth.euler_experiment import EulerExperiment
from morinth.quadrature import GaussLegendre
from morinth.boundary_conditions import Outflow

class EntropyShockIC(object):
    def __init__(self, model):
        self.model = model
        self.amplitude = 0.01
        self.wave_number = 13
        self.x_crit = 0.5
        self.quadrature = GaussLegendre(5)

    def __call__(self, grid):
        x = grid.cell_centers
        w = np.empty((4,) + x.shape[:1])
        is_inside = x[:,0] < self.x_crit

        w[0,...] = np.where(is_inside,
                            3.85714,
                            np.exp(-self.amplitude*np.sin(self.wave_number*x[:,0])))
        w[1,...] = np.where(is_inside, 2.629369, 0.0)
        w[2,...] = 0.0
        w[3,...] = np.where(is_inside, 10.33333, 1.0)

        return self.model.conserved_variables(w)


class EntropyShock(EulerExperiment):
    @property
    def final_time(self):
        return 2.0

    @property
    def n_cells(self):
        return 200

    @property
    def domain(self):
        return [0.0, 5.0]

    @property
    def order(self):
        return 5

    @property
    def initial_condition(self):
        return EntropyShockIC(self.model)

    @property
    def output_filename(self):
        return "img/entropy_shock"

    @property
    def boundary_condition(self):
        return Outflow(self.grid)


if __name__ == '__main__':
    sim = EntropyShock()
    sim()
