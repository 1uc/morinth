#!/usr/bin/env python
# encoding: utf-8

# Cloud-shock interaction from [Mishra, Schwab, Sukys 2012].

import numpy as np

from boundary_conditions import Outflow
from euler_experiment import EulerExperiment2D

class CloudShockIC:
    def __init__(self, model):
        self.model = model
        self.Y = np.array(8*[1.0])
        self.Y[0] = 1.0/25.0 + self.Y[0]/50.0

    def __call__(self, grid):
        model, Y = self.model, self.Y

        xy = grid.cell_centers
        u0 = np.zeros((4,) + xy.shape[:2])

        # background state
        u0[0,...] = 1.0
        u0[3,...] = 1.0

        # shock
        is_inside = xy[:,:,0] < Y[0]
        u0[0,is_inside] = 3.86859 + 1.0/10.0*Y[6]
        u0[1,is_inside] = 11.2536
        u0[2,is_inside] = 0.0
        u0[3,is_inside] = 167.345 + Y[7]

        # cloud
        r = np.sqrt((xy[...,0] - 0.25)**2 + (xy[...,1] - 0.5)**2)
        is_inside = r < self.r_crit(xy, r)

        rhoA = 10 + 0.5*Y[1] + Y[2]*np.sin(4.0*(xy[...,0] - 0.25))
        rhoB = 0.5*Y[3]*np.cos(8.0*(xy[...,1] - 0.5))
        rho0 = rhoA + rhoB

        u0[0,...] = np.where(is_inside, rho0, u0[0,...])

        return self.model.conserved_variables(u0)

    def r_crit(self, xy, r):
        Y = self.Y

        r_min = 0.13

        theta = (xy[...,0] - 0.25)/r
        r_crit = r_min + 1.0/50*Y[5]*np.sin(theta) + 1.0/100.0*Y[5]*np.sin(10.0*theta)
        r_crit = np.where(r < r_min, r_min, r_crit)

        return r_crit

class CloudShock(EulerExperiment2D):
    """Interaction of a travelling shock with a stationary high-density cloud."""

    @property
    def final_time(self):
        return 0.06

    @property
    def n_cells(self):
        return 100

    @property
    def order(self):
        return 1

    @property
    def initial_condition(self):
        return CloudShockIC(self.model)

    @property
    def boundary_condition(self):
        return Outflow(self.grid)

    @property
    def output_filename(self):
        return "img/cloud_shock"

    @property
    def steps_per_frame(self):
        return 5


if __name__ == '__main__':
    cloud_shock = CloudShock()
    cloud_shock()
