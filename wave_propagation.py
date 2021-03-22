#! /usr/bin/env python3
# -*- encoding: utf-8 -*-

import numpy as np

from morinth.gaussian_bump import EquilibriumExperiment
from morinth.gaussian_bump import EquilibriumConvergenceRates
from morinth.boundary_conditions import PeriodicVelocity
from morinth.euler import LinearGravity
from morinth.time_keeper import PlotLast, PlotNever, PlotEveryNthStep
from morinth.runge_kutta import *

class WavePropagationIC():
    def __init__(self, model, equilibrium):
        self.model = model
        self.equilibrium = equilibrium

        self.x_ref = 0.0
        self.rho_ref = 1.0
        self.p_ref = 1.0
        self.T_ref = self.model.temperature(p=self.p_ref, rho=self.rho_ref)
        E_int_ref = model.internal_energy(p=self.p_ref)
        self.u_ref = np.array([self.rho_ref, 0.0, 0.0, E_int_ref])

    def __call__(self, grid):
        return self.back_ground(grid)

    def back_ground(self, grid):
        x = grid.cell_centers[:,0]

        equilibrium = self.equilibrium
        u_bar_ref = equilibrium.cell_averages(self.u_ref, self.x_ref)

        return equilibrium.reconstruct(u_bar_ref, self.x_ref, x)


class WavePropagation(EquilibriumExperiment):
    @property
    def final_time(self):
        # return 0.125 + 0.0625
        return 0.1

    @property
    def gamma(self):
        return 5.0/3.0

    @property
    def gravity(self):
        return LinearGravity(1.0)

    @property
    def initial_condition(self):
        equilibrium = self.thermodynamic_equilibrium(self.grid, self.model)
        return WavePropagationIC(self.model, equilibrium)

    @property
    def boundary_condition(self):
        equilibrium = self.thermodynamic_equilibrium(self.grid, self.model)
        bc = PeriodicVelocity(self.grid, equilibrium)
        bc.amplitude = self.wave_amplitude

        return bc

    @property
    def domain(self):
        return [0.0, 2.0]

    @property
    def output_filename(self):
        pattern = self.base_filename + "_{:s}_{:s}_res{:05d}"
        return pattern.format(self.well_balancing,
                              self._thermodynamic_equilibrium,
                              self.n_cells)

    @property
    def plotting_steps(self):
        return PlotNever()
        # return PlotEveryNthStep(10)


class StrongWavePropagation(WavePropagation):
    @property
    def wave_amplitude(self):
        return 1e-1

    @property
    def base_filename(self):
        return "wave_propagation_strong"

    @property
    def output_filename(self):
        return self.base_filename + "-" + self.well_balancing


class StrongWavePropagationReference(StrongWavePropagation):
    @property
    def well_balancing(self):
        return "wb_o4"

    @property
    def n_cells(self):
        return 2**13 + 6

    @property
    def output_filename(self):
        pattern = "strong_wave_propagation-{:s}"
        return pattern.format(self._thermodynamic_equilibrium)


class WeakWavePropagation(WavePropagation):
    @property
    def wave_amplitude(self):
        return 1e-8

    @property
    def base_filename(self):
        return "wave_propagation_weak"


class WeakWavePropagationReference(WeakWavePropagation):
    @property
    def well_balancing(self):
        return "wb_o2"

    @property
    def n_cells(self):
        return 2**11 + 6

    @property
    def output_filename(self):
        pattern = "weak_wave_propagation-{:s}"
        return pattern.format(self._thermodynamic_equilibrium)


class WavePropagationRates(EquilibriumConvergenceRates):
    def __init__(self):
        super().__init__()

        self.all_labels = ["$\\rho\,v_{1,(0)}$", "$E_{(0)}$",
                           "$\\rho\,v_{1,(1)}$", "$E_{(1)}$",
                           "$\\rho\,v_{1,(2)}$", "$E_{(2)}$"]

        self.error_vars = [1, 3]
        self.resolutions = 2**np.arange(7, 10) + 6

        # self.is_reference_solution_required = True
        self.is_reference_solution_required = False


if __name__ == "__main__":
    sim = WavePropagationRates()

    sim(WeakWavePropagation, WeakWavePropagationReference, "isentropic")
    sim(StrongWavePropagation, StrongWavePropagationReference, "isentropic")
