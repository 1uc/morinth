#! /usr/bin/env python3
# -*- encoding: utf-8 -*-

# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

import numpy as np

from gaussian_bump import EquilibriumExperiment
from gaussian_bump import EquilibriumConvergenceRates
from boundary_conditions import Periodic
from euler import LinearGravity
from time_keeper import PlotLast, PlotNever, PlotEveryNthStep
from runge_kutta import *
from quadrature import GaussLegendre

class SmoothWaveIC():
    def __init__(self, model, equilibrium=None):
        self.model = model

        self.x_ref = 0.0
        self.rho_ref = 1.0
        self.p_ref = 1.0
        self.T_ref = self.model.temperature(p=self.p_ref, rho=self.rho_ref)
        E_int_ref = model.internal_energy(p=self.p_ref)
        self.u_ref = np.array([self.rho_ref, 0.0, 0.0, E_int_ref])

    def __call__(self, grid):
        cell_averages = GaussLegendre(10)
        return self.back_ground(grid) + cell_averages(self.delta, grid)

    def back_ground(self, grid):
        x = grid.cell_centers[:,0]
        u = np.empty_like((4,) +  x.shape)
        u[:,:] = self.u_ref[:,np.newaxis]
        return u

    def delta(self, x):
        du = np.zeros_like((4,) + x)
        du[3,...] = amplitude*np.sin(2.0*np.pi*x)

        return du
