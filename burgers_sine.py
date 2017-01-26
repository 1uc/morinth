#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
from burgers import Burgers

from grid import Grid
from finite_volume_fluxes import scheme_o1, scheme_weno
from visualize import SimpleGraph
from boundary_conditions import Periodic
from euler_experiment import BurgersExperiment

class BurgersSineWave(BurgersExperiment):
    @property
    def n_cells(self):
        return 1000

    @property
    def order(self):
        return 5

    @property
    def steps_per_frame(self):
        return 10

    @property
    def output_filename(self):
        return "img/burgers_sine_wave"

    @property
    def final_time(self):
        return 2.0

    @property
    def initial_condition(self):
        return lambda grid: np.cos(4.0*np.pi*grid.cell_centers).reshape((1, -1))

    @property
    def boundary_condition(self):
        return Periodic(self.grid)


if __name__ == '__main__':
    burgers_sine_wave = BurgersSineWave()
    burgers_sine_wave()

