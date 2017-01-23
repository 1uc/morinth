#!/usr/bin/env python
# encoding: utf-8

import numpy as np
from burgers import Burgers

from grid import Grid
from weno import WENO
from finite_volume_fluxes import scheme_o1, scheme_weno
from visualize import SimpleGraph
from time_keeper import FixedDuration, PlotEveryNthStep
from progress_bar import ProgressBar
from boundary_conditions import Periodic
from time_loop import TimeLoop

def burgers_sine_wave():
    model = Burgers()
    grid = Grid([0.0, 1.0], 300, 3)
    bc = Periodic(grid)
    _, _, single_step = scheme_weno(model, grid, bc)

    visualize = SimpleGraph(grid, "img/burgers_sine_wave")
    plotting_steps = PlotEveryNthStep(10)
    progress_bar = ProgressBar(10)

    time_keeper = FixedDuration(4.0, needs_baby_steps=True)
    simulation = TimeLoop(single_step, visualize, plotting_steps, progress_bar)

    u0 = np.sin(4.0*np.pi*grid.cell_centers).reshape((1, -1))
    simulation(u0, time_keeper)

if __name__ == '__main__':
    burgers_sine_wave()

