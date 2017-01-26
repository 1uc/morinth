#!/usr/bin/env python3
# encoding: utf-8

import numpy as np

from boundary_conditions import Outflow
from weno import StableWENO, OptimalWENO, PrimitiveReconstruction
from rusanov import Rusanov

from euler_experiment import EulerExperiment

class ShockTubeIC:
    """Toro's shock-tube."""

    def __init__(self, model):
        self.model = model

    def __call__(self, grid):
        u0 = np.empty((4, grid.cell_centers.shape[0]))

        x_crit = 0.3
        is_inside = grid.cell_centers[...,0] < x_crit

        u0[0, ...] = np.where(is_inside, 1.0, 0.125)
        u0[1, ...] = np.where(is_inside, 0.75, 0.0)
        u0[2, ...] = 0.0
        u0[3, ...] = np.where(is_inside, 1.0, 0.1)

        return self.model.conserved_variables(u0)


class ShockTubeBase(EulerExperiment):
    @property
    def final_time(self):
        return 0.2

    @property
    def n_cells(self):
        return 100

    @property
    def flux(self):
        return Rusanov(self.model)

    @property
    def steps_per_frame(self):
        return 30

    @property
    def boundary_condition(self):
        return Outflow(self.grid)

    @property
    def initial_condition(self):
        return ShockTubeIC(self.model)

    @property
    def needs_baby_steps(self):
        return True


class ShockTubeO1(ShockTubeBase):
    @property
    def output_filename(self):
        return "img/shock_tube-o1"

    @property
    def order(self):
        return 1


class ENOShockTube(ShockTubeBase):
    @property
    def output_filename(self):
        return "img/shock_tube-eno"

    @property
    def order(self):
        return 3


class WENOShockTube(ShockTubeBase):
    @property
    def output_filename(self):
        return "img/shock_tube-weno"

    @property
    def reconstruction(self):
        return OptimalWENO()

    @property
    def order(self):
        return 5

if __name__ == '__main__':
    # all_solvers = [WENOShockTube()]
    all_solvers = [ShockTubeO1(), ENOShockTube(), WENOShockTube()]

    for shock_tube in all_solvers:
         shock_tube()
