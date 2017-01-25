#!/usr/bin/env python3
# encoding: utf-8
import numpy as np

from euler_experiment import EulerExperiment
from finite_volume_fluxes import scheme_o1, scheme_weno, scheme_eno
from boundary_conditions import Periodic

class PeriodicEulerExperiment(EulerExperiment):
    @property
    def boundary_condition(self):
        return Periodic(self.grid)

def gaussian(x, sigma):
    return np.exp(-(x/sigma)**2)


class SoundWaveIC(object):
    """Small smooth pressure bump."""

    def __init__(self, model):
        self.model = model

    def __call__(self, grid):
        x = grid.cell_centers
        mean, sigma, offset = 0.1, 0.1, 0.5

        u0 = np.empty((4,) + x.shape)
        u0[0,...] = 1.0
        u0[1,...] = 0.0
        u0[2,...] = 0.0
        u0[3,...] = 1.0 + mean*gaussian(x - offset, sigma)

        return self.model.conserved_variables(u0)


class SoundWave(PeriodicEulerExperiment):
    """Sound wave in a periodic box."""

    @property
    def n_cells(self):
        return 200

    @property
    def final_time(self):
        return 1.0

    @property
    def initial_condition(self):
        return SoundWaveIC(self.model)

    @property
    def output_filename(self):
        return "img/sound_wave"

    @property
    def steps_per_frame(self):
        return 5

    @property
    def order(self):
        return 5

if __name__ == '__main__':
    sound_wave = SoundWave()
    sound_wave()
