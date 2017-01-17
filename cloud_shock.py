#!/usr/bin/env python
# encoding: utf-8

# Cloud-shock interaction from [Mishra, Schwab, Sukys 2012].

import numpy as np
from euler import Euler
from hllc import HLLC
from grid import Grid
from visualize import EulerColormaps
from time_loop import TimeLoop
from time_keeper import PlotEveryNthStep, FixedDuration
from boundary_conditions import Outflow

from euler_experiment import EulerExperiment, scheme_o1

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

class CloudShock(EulerExperiment):

    def __call__(self):
        ic = CloudShockIC(self.model)
        u0 = ic(self.grid)

        _, _, single_step = scheme_o1(self.model, self.grid, self.boundary_conditions)

        time_keeper = FixedDuration(0.06)
        simulation = TimeLoop(single_step, self.visualize, self.plotting_steps)
        simulation(u0, time_keeper)

    def set_up_grid(self):
        self.grid = Grid([[0.0, 1.0], [0.0, 1.0]], (800, 800), 1)

    def set_up_visualization(self):
        self.visualize = EulerColormaps(self.grid, "img/cloud_shock", self.model)
        self.plotting_steps = PlotEveryNthStep(steps_per_frame = 10)

    def set_up_boundary_condition(self):
        self.boundary_conditions = Outflow(self.grid)


if __name__ == '__main__':
    cloud_shock = CloudShock()
    cloud_shock()
