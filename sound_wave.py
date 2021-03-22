#!/usr/bin/env python3
# encoding: utf-8
import numpy as np

from morinth.euler_experiment import EulerExperiment
from morinth.weno import OptimalWENO
from morinth.boundary_conditions import Periodic
from morinth.math_tools import gaussian

class SoundWaveIC(object):
    """Small smooth pressure bump."""

    def __init__(self, model):
        self.model = model

    def __call__(self, grid):
        x = grid.cell_centers[:,0]
        mean, sigma, offset = 0.1, 0.1, 0.5

        u0 = np.empty((4, x.shape[0]))
        u0[0,...] = 1.0
        u0[1,...] = 0.0
        u0[2,...] = 0.0
        u0[3,...] = 1.0 + mean*gaussian(x - offset, sigma)

        return self.model.conserved_variables(u0)


class AdvectiveIC(object):

    def __init__(self, model):
        self.model = model

    def __call__(self, grid):
        x = grid.cell_centers[:,0]

        u0 = np.empty((4, x.shape[0]))
        u0[0,...] = 1.0
        u0[1,...] = 0.3*np.sin(2*np.pi*x)
        u0[2,...] = 0.0
        u0[3,...] = 1.0

        return self.model.conserved_variables(u0)


class PeriodicEulerExperiment(EulerExperiment):
    @property
    def boundary_condition(self):
        return Periodic(self.grid)

    @property
    def order(self):
        return 5


class SoundWave(PeriodicEulerExperiment):
    """Sound wave in a periodic box."""

    @property
    def n_cells(self):
        return 200

    @property
    def final_time(self):
        return 3.0

    @property
    def initial_condition(self):
        return SoundWaveIC(self.model)

    @property
    def output_filename(self):
        return "img/sound_wave"

    @property
    def steps_per_frame(self):
        return 30


class SmoothAdvection(PeriodicEulerExperiment):
    """Advection test-case."""

    @property
    def final_time(self):
        return 3.0

    @property
    def n_cells(self):
        return 200

    @property
    def initial_condition(self):
        return AdvectiveIC(self.model)

    @property
    def output_filename(self):
        return "img/smooth_advection"

    @property
    def steps_per_frame(self):
        return 10



if __name__ == '__main__':
    all_experiments = [SoundWave(), SmoothAdvection()]

    for experiment in all_experiments:
        experiment()
