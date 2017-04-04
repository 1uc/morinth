#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pickle

from equilibrium import IsothermalEquilibrium, IsentropicEquilibrium
from euler_experiment import EulerExperiment
from boundary_conditions import Outflow, IsothermalOutflow
from visualize import EquilibriumGraphs, DensityGraph, ConvergencePlot
from weno import OptimalWENO, EquilibriumStencil
from source_terms import BalancedSourceTerm
from math_tools import gaussian, l1_error, convergence_rate
from time_keeper import FixedSteps, PlotNever, PlotLast
from quadrature import GaussLegendre
from coding_tools import with_default
from latex_tables import LatexConvergenceTable


class GaussianBumpIC(object):
    def __init__(self, model, equilibrium):
        self.model = model
        self.equilibrium = equilibrium

        self.p_amplitude = 1.0e-5
        self.rho_amplitude = 1.0e-5
        self.sigma = 0.05

        self.x_ref = 0.0
        self.rho_ref = 2.0
        self.p_ref = 1.0
        self.T_ref = self.model.temperature(p=self.p_ref, rho=self.rho_ref)
        E_int_ref = model.internal_energy(p=self.p_ref)
        self.u_ref = np.array([self.rho_ref, 0.0, 0.0, E_int_ref])

        self.x_mid = 0.5
        self.p_mid = 0.3

        self.quadrature = GaussLegendre(5)

    def __call__(self, grid):
        du_lambda = lambda x : self.delta(x, as_conserved_variables=True)
        u0 = self.back_ground(grid)
        du0 = self.quadrature(grid.edges, du_lambda)

        assert np.all(np.isfinite(u0))
        assert np.all(np.isfinite(du0))

        return u0 + du0

    def delta(self, x, as_conserved_variables=False):
        """Perturbation in respective variables."""

        dp = self.p_mid*gaussian(x - self.x_mid, self.sigma)
        drho = self.model.rho(p=dp, T=self.T_ref)

        dw = np.zeros((4,) + dp.shape)
        dw[0,...] = self.rho_amplitude * drho
        dw[3,...] = self.p_amplitude * dp

        if as_conserved_variables:
            dw[3,...] = self.model.internal_energy(p=dw[3,...])

        return dw

    def back_ground(self, grid):
        x = grid.cell_centers[:,0]
        equilibrium = self.equilibrium
        u_bar_ref = equilibrium.cell_averages(self.u_ref, self.x_ref)

        return equilibrium.reconstruct(u_bar_ref, self.x_ref, x)

    def point_values(self, x):
        """Point values of the primitive variables."""

        equilibrium = self.equilibrium
        rho_ref, p_ref, x_ref = self.rho_ref, self.p_ref, self.x_ref

        w = np.zeros((4,) + x.shape)
        w[0,...], w[3,...] = equilibrium.extrapolate(rho_ref, p_ref, x_ref, x)
        dw = self.delta(x)

        return w + dw


class GaussianBump(EulerExperiment):

    @property
    def final_time(self):
        return 0.2

    @property
    def n_cells(self):
        return getattr(self, "_n_cells", 200)

    @n_cells.setter
    def n_cells(self, new_value):
        self._n_cells = new_value

    @property
    def initial_condition(self):
        equilibrium = self.thermodynamic_equilibrium(self.grid, self.model)
        ic = GaussianBumpIC(self.model, equilibrium)
        ic.p_amplitude, ic.rho_amplitude = 1e-5, 1e-5

        return ic

    @property
    def order(self):
        return 5

    @property
    def gravity(self):
        return 1.0

    @property
    def boundary_condition(self):
        return IsothermalOutflow(self.grid, self.model)

    @property
    def weno(self):
        mode = self.well_balancing

        if mode == "wb_o2" or mode == "wb_o4":
            return OptimalWENO(EquilibriumStencil(self.grid, self.equilibrium, self.model))

        elif mode == "naive":
            return OptimalWENO()

        else:
            raise Exception("Wrong `well_balancing`.")

    @property
    def thermodynamic_equilibrium(self):
        return self._thermodynamic_equilibrium

    @thermodynamic_equilibrium.setter
    def thermodynamic_equilibrium(self, rhs):
        self._thermodynamic_equilibrium = rhs

    @property
    def source(self):
        mode = self.well_balancing

        if mode == "wb_o2" or mode == "wb_o4":
            source_order = 4.0 if mode == "wb_o4" else 2.0
            return BalancedSourceTerm(self.grid, self.model,
                                      self.equilibrium, source_order)

        elif mode == "naive":
            return super().source

        else:
            raise Exception("Wrong `well_balancing`.")

    @property
    def well_balancing(self):
        return self._well_balancing

    @well_balancing.setter
    def well_balancing(self, rhs):
        self._well_balancing = rhs

    @property
    def equilibrium(self):
        mode = self.well_balancing
        if mode == "wb_o2" or mode == "wb_o4":
            return self.thermodynamic_equilibrium(self.grid, self.model)

        elif mode == "naive":
            return None

        else:
            assert Exception("Wrong `well_balancing`.")

    @property
    def visualize(self):
        back_ground = self.initial_condition.back_ground(self.grid)
        return EquilibriumGraphs(self.grid, self.output_filename, self.model, back_ground)

    @property
    def output_filename(self):
        return "img/gaussian_bump"

    @property
    def steps_per_frame(self):
        return 5

class GaussianBumpConvergence(GaussianBump):
    @property
    def plotting_steps(self):
        return PlotLast()

    @property
    def output_filename(self):
        return "img/gaussian_bump_{:05d}".format(int(self.n_cells))

class GaussianBumpReference(GaussianBumpConvergence):
    @property
    def well_balancing(self):
        return "wb_o4"

    @property
    def n_cells(self):
        return 2**13 + 6


def compute_reference_solution(Equilibrium, label):
    gaussian_bump = GaussianBumpReference()
    gaussian_bump.thermodynamic_equilibrium = Equilibrium
    grid = gaussian_bump.grid

    u0 = gaussian_bump.initial_condition.back_ground(grid)
    u_ref = gaussian_bump()

    filename_base = "data/gaussian_bump-{:s}".format(label)
    np.save(filename_base + "_background.npy", u0)
    np.save(filename_base + "_reference.npy", u_ref)

    with open(fileame_base + "_grid.pkl", 'wb') as f:
        pickle.dump(grid, f)

    return u0, u_ref, grid

def load_reference_solution(label):
    filename_base = "data/gaussian_bump-{:s}".format(label)
    u0_ref = np.load(filename_base + "_background.npy")
    u_ref = np.load(filename_base + "_reference.npy")

    with open(filename_base + "_grid.pkl", 'rb') as f:
        grid = pickle.load(f)

    return u0_ref, u_ref, grid

def down_sample(u_fine, grid_fine, grid_coarse):
    """Compute cell-averages of `u_fine` on the coarse grid."""
    if grid_fine.n_dims == 2:
        raise Exception("Needs to be implemented.")

    ngf = grid_fine.n_ghost
    ngc = grid_coarse.n_ghost

    ncf = grid_fine.n_cells[0] - 2*ngf
    ncc = grid_coarse.n_cells[0] - 2*ngc
    r = ncf // ncc
    assert r*ncc == ncf

    shape = (u_fine.shape[0], -1, r)
    u_coarse = np.mean(u_fine[:,ngf:-ngf].reshape(shape), axis=-1)

    return u_coarse

def compute_convergence(Equilibrium, label, well_balancing):
    u0_ref, u_ref, grid_ref = load_reference_solution(label)
    du_ref = u_ref - u0_ref

    resolutions = 2**np.arange(3, 11) + 6

    err = np.empty((4, resolutions.size))
    for l, res in enumerate(resolutions):
        gaussian_bump = GaussianBumpConvergence()
        gaussian_bump.thermodynamic_equilibrium = Equilibrium
        gaussian_bump.well_balancing = well_balancing
        gaussian_bump.n_cells = res
        grid = gaussian_bump.grid
        n_ghost = grid.n_ghost

        u0 = gaussian_bump.initial_condition.back_ground(grid)
        u = gaussian_bump()

        du = u - u0

        ic = gaussian_bump.initial_condition
        du_ref_c = down_sample(du_ref, grid_ref, grid)
        err[:,l] = l1_error(du[:,n_ghost:-n_ghost], du_ref_c)

    rho_rate = convergence_rate(err[0,...], resolutions-6)
    E_int_rate = convergence_rate(err[3,...], resolutions-6)

    return [err[0,...], err[3,...]], [rho_rate, E_int_rate], resolutions


def run_compute_convergence(Equilibrium, label):
    all_errors, all_rates = [], []
    all_labels = ["$\\rho_{(0)}$", "$E_{(0)}$",
                  "$\\rho_{(1)}$", "$E_{(1)}$",
                  "$\\rho_{(2)}$", "$E_{(2)}$"]

    compute_reference_solution(Equilibrium, label)
    for well_balancing in ["naive", "wb_o2", "wb_o4"]:
        error, rate, resolutions = compute_convergence(Equilibrium, label, well_balancing)
        all_errors += error
        all_rates += rate

    filename_base = "img/code-validation/gaussian_bump-{:s}".format(label)
    latex_table = LatexConvergenceTable(all_errors, all_rates, resolutions-6, all_labels)
    latex_table.write(filename_base + ".tex")

    plot = ConvergencePlot([4, 5])
    plot(all_errors, resolutions-6, all_labels)
    plot.save(filename_base)

if __name__ == "__main__":
    # run_compute_convergence(IsothermalEquilibrium, "isothermal")
    run_compute_convergence(IsentropicEquilibrium, "isentropic")
