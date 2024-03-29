# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

import numpy as np
import matplotlib.pyplot as plt

from morinth.source_terms import BalancedSourceTerm, EquilibriumDensityInterpolation, UnbalancedSourceTerm
from morinth.grid import Grid
from morinth.euler import Euler, PointMassGravity
from morinth.equilibrium import IsothermalEquilibrium, IsentropicEquilibrium
from morinth.weno import OptimalWENO, EquilibriumStencil
from morinth.gaussian_bump import GaussianBumpIC
from morinth.quadrature import GaussLegendre
from morinth.math_tools import l1_error, linf_error, convergence_rate
from morinth.latex_tables import LatexConvergenceTable
from morinth.visualize import ConvergencePlot

import morinth.testing_tools as testing_tools
from morinth.testing_tools import pytest_config

def check_source_term_order(order, Equilibrium):
    model = Euler(gamma=1.4, gravity = 1.2, specific_gas_constant=2.0)
    quadrature = GaussLegendre(5)
    all_resolutions = 2**np.arange(3, 11) + 6

    err = np.empty((4, all_resolutions.size))

    for l, res in enumerate(all_resolutions):
        grid = Grid([0.0, 1.0], res, 3)
        equilibrium = Equilibrium(grid, model)
        ic = GaussianBumpIC(model, equilibrium)
        ic.p_amplitude, ic.rho_amplitude = 0.0, 0.1

        weno = OptimalWENO(EquilibriumStencil(grid, equilibrium, model))
        source_term = BalancedSourceTerm(grid, model, equilibrium, order=order)

        u_bar = ic(grid)
        u_plus, u_minus = weno(u_bar, axis=0)
        u_left, u_right = u_minus[:,:-1,...], u_plus[:,1:,...]
        s_approx = source_term.edge_source(u_bar, u_left, u_right, axis=0)

        S_momentum = lambda x: -ic.point_values(x)[0,...]*model.gravity.dphi_dx(x)

        s_ref = np.zeros_like(s_approx)
        s_ref[1,...] = quadrature(grid.edges[3:-3,...], S_momentum)

        err[:, l] = linf_error(s_approx, s_ref)

    rate = convergence_rate(err[1, ...], all_resolutions-6)
    return err[1,...], rate, all_resolutions

def source_term_order(Equilibrium, label):
    all_errors, all_rates = [], []
    all_labels = ["$S_{(1)}$", "$S_{(2)}$"]

    for order in [2, 4]:
        errors, rates, resolutions = check_source_term_order(order, Equilibrium)
        all_errors.append(errors)
        all_rates.append(rates)

    filename_base = "img/code-validation/source_term-{:s}".format(label)
    table = LatexConvergenceTable(all_errors, all_rates, resolutions-6, all_labels)
    table.write(filename_base + ".tex")

    plot = ConvergencePlot(trend_orders=[2, 4])
    plot(all_errors, resolutions-6, all_labels)
    plot.save(filename_base)

    assert np.abs(np.max(all_rates[1]) - 4.0) < 0.1

def test_source_term_order():
    source_term_order(IsothermalEquilibrium, "isothermal")
    source_term_order(IsentropicEquilibrium, "isentropic")


def test_unbalanced_source_term():
    gravity = PointMassGravity(1.0, 1.0, 1.0)
    model = Euler(gamma = 1.4, gravity = gravity, specific_gas_constant = 1.0)

    def rho(x):
        return 2.0 + np.sin(3*np.pi*x) * np.cos(2*np.pi*x)

    def dphi_dx(x):
        return model.gravity.dphi_dx(x)

    quadrature = GaussLegendre(5)

    all_resolutions = 2**np.arange(4, 10) + 6
    err = np.empty((1, all_resolutions.size))

    for l, res in enumerate(all_resolutions):
        grid = Grid([0.0, 1.0], res, 3)

        def ic(x):
            u0 = np.zeros((4,) + x.shape)
            u0[0,...] = rho(x)
            u0[3,...] = 1.0

            return u0

        u_bar = quadrature(grid.edges, ic)

        source_term = UnbalancedSourceTerm(grid, model)
        s_approx = source_term.volume_source(u_bar)
        s_ref = quadrature(grid.edges[2:-2,...], lambda x : -rho(x)*dphi_dx(x))

        err[:, l] = l1_error(s_approx[1,2:-2,...], s_ref)

    rate = convergence_rate(err, all_resolutions-6)
    assert np.all(np.abs(rate - 4.0) < 0.15)

def test_equilibrium_interpolation(pytest_config):
    n_ghost = 3
    model = Euler(gamma = 1.4, gravity = 1.0, specific_gas_constant = 1.0)

    alpha = 0.25

    resolutions = 2**np.arange(3, 10) + 6
    err = np.empty(resolutions.size)

    filename_base = "img/code-validation/equilibrium_interpolation-{:d}"

    for l, n_cells in enumerate(resolutions):
        grid = Grid([0.0, 1.0], n_cells, n_ghost)
        equilibrium = IsothermalEquilibrium(grid, model)
        ic = GaussianBumpIC(model, equilibrium)
        ic.rho_amplitude, ic.p_amplitude = 1.0, 0.0
        stencil = EquilibriumStencil(grid, equilibrium, model)

        x0 = grid.edges[n_ghost:-n_ghost-1,0]
        x1 = grid.edges[n_ghost+1:-n_ghost,0]
        x_ref = grid.cell_centers[n_ghost:-n_ghost,0]

        u0 = ic(grid)
        interpolate = EquilibriumDensityInterpolation(grid, model, equilibrium, u0)
        x = x0 + alpha*(x1 - x0)

        w_exact = ic.point_values(x)
        rho_exact = w_exact[0,...]

        rho_ref, p_ref = equilibrium.point_values(u0[:,3:-3], x_ref)
        rho_eq_approx, _ = equilibrium.extrapolate(rho_ref, p_ref, x_ref, x)
        drho_approx = interpolate(alpha)
        rho_approx = rho_eq_approx + drho_approx

        plt.clf()
        plt.plot(x_ref, u0[0,3:-3], label="ref")
        plt.plot(x, rho_eq_approx, label="loc. eq")
        plt.plot(x, rho_approx, label="approx")
        plt.plot(x, drho_approx, label="delta")
        plt.legend()

        filename = filename_base.format(l)
        plt.savefig(filename + ".eps")
        plt.savefig(filename + ".png")

        if testing_tools.is_manual_mode(pytest_config):
            plt.show()

        err[l] = l1_error(rho_approx[np.newaxis,:], rho_exact[np.newaxis,:])

    rate = convergence_rate(err, resolutions-6)
    assert np.all(np.abs(rate[3:] - 5.0) < 0.2)
