import numpy as np

from weno import OptimalWENO, EquilibriumStencil
from grid import Grid
from quadrature import GaussLegendre
from euler import Euler
from equilibrium import IsothermalEquilibrium
from gaussian_bump import GaussianBumpIC
from latex_tables import LatexConvergenceTable

import matplotlib.pylab as plt

import pytest
import testing_tools
from math_tools import convergence_rate, l1_error, linf_error
from visualize import ConvergencePlot

def test_equilibrium_order():
    model = Euler(gamma = 1.4, gravity = 1.0, specific_gas_constant = 1.0)
    all_resolutions = 2**np.arange(4, 8)

    ic = GaussianBumpIC(model)
    ic.p_amplitude, ic.rho_amplitude = 0.0, 1.0

    quadrature = GaussLegendre(5)
    err = np.empty((4, all_resolutions.size))

    for l, res in enumerate(all_resolutions):
        grid = Grid([0.0, 1.0], res, 0)

        approximate = ic(grid)

        ic_point_values = lambda x: model.conserved_variables(ic.point_values(x))
        reference = quadrature(grid.edges, ic_point_values)
        err[:,l] = l1_error(approximate, reference)

    rate = convergence_rate(err[[0,3],...], all_resolutions)
    assert np.all(np.abs(rate - 4.0) < 0.1)

def test_weno_well_balanced():
    model = Euler(gamma = 1.4, gravity = 1.0, specific_gas_constant = 1.0)
    resolutions = np.array([16, 32, 64, 128]).reshape((-1,1))
    p_ref, T_ref, x_ref = 1.0, 2.0, 0.0

    ic = GaussianBumpIC(model)
    ic.p_amplitude, ic.rho_amplitude = 0.0, 0.0

    for resolution in resolutions:
        plt.clf()

        grid = Grid([0.0, 1.0], resolution, 3)
        equilibrium = IsothermalEquilibrium(grid, model)
        weno = OptimalWENO(EquilibriumStencil(grid, equilibrium, model))

        u0 = ic(grid)
        u_plus, u_minus = weno(u0, axis=0)
        assert np.all(np.abs(u_plus - u_minus) < 1e-13)

        if testing_tools.is_manual_mode():
            plt.plot(grid.edges[3:-3,0], u_plus[0,:], '>')
            plt.plot(grid.cell_centers[3:-3,0], u0[0,3:-3], 'k_')
            plt.plot(grid.edges[3:-3,0], u_minus[0,:], '<')
            plt.show()

def test_weno_well_balanced_order():
    model = Euler(gamma = 1.4, gravity = 1.0, specific_gas_constant = 1.0)
    resolutions = 2**np.arange(4, 10)

    ic = GaussianBumpIC(model)
    ic.p_amplitude, ic.rho_amplitude = 1e-5, 1e-5

    err_plus = np.empty((4, len(resolutions)))
    err_minus = np.empty((4, len(resolutions)))

    for l, resolution in enumerate(np.nditer(resolutions)):
        grid = Grid([0.0, 1.0], resolution, 3)
        equilibrium = IsothermalEquilibrium(grid, model)
        weno = OptimalWENO(EquilibriumStencil(grid, equilibrium, model))

        u0 = ic(grid)
        u_plus, u_minus = weno(u0, axis=0)
        w_plus = model.primitive_variables(u_plus)
        w_minus = model.primitive_variables(u_minus)

        x = grid.edges[3:-3,...,0]
        w_exact = ic.point_values(x)

        err_plus[:,l] = linf_error(w_plus, w_exact)
        err_minus[:,l] = linf_error(w_minus, w_exact)

    rho_rate_plus = convergence_rate(err_plus[0,...], resolutions-6)
    rho_rate_minus = convergence_rate(err_minus[0,...], resolutions-6)

    p_rate_plus = convergence_rate(err_plus[3,...], resolutions-6)
    p_rate_minus = convergence_rate(err_minus[3,...], resolutions-6)

    def achieves(observed_rate, expected):
        assert np.all(np.abs(observed_rate[-2:] - expected) < 0.1)

    achieves(rho_rate_plus, 5.0)
    achieves(rho_rate_minus, 5.0)
    achieves(p_rate_plus, 5.0)
    achieves(p_rate_minus, 5.0)

    all_errors = [err_plus[0,:], err_minus[0,:], err_plus[3,:], err_minus[3,:]]
    all_rates = [rho_rate_plus, rho_rate_minus, p_rate_plus, p_rate_minus]
    all_labels = ["$\\rho^+$", "$\\rho^-$", "$p^+$", "$p^-$"]

    filename_base = "img/code-validation/weno_well-balanced"

    table = LatexConvergenceTable(all_errors, all_rates, resolutions, all_labels)
    table.write(filename_base + ".tex")

    plot = ConvergencePlot(trend_orders=[5])
    plot(all_errors, resolutions, all_labels)
    plot.save(filename_base)

def test_weno_in_equiblibrium():
    model = Euler(gamma = 1.4, gravity = 1.0, specific_gas_constant = 1.0)
    grid = Grid([0.0, 1.0], 16, 3)
    equilibrium = IsothermalEquilibrium(grid, model)

    ic = GaussianBumpIC(model)
    ic.p_amplitude, ic.rho_amplitude = 0.0, 0.0
    rho_ref = ic.rho_ref
    p_ref = ic.p_ref
    T_ref = model.temperature(rho=rho_ref, p=p_ref)
    x_ref = ic.x_ref

    u0 = ic(grid)

    x = grid.cell_centers[...,0]
    rho_exact, p_exact = equilibrium.extrapolate(p_ref, T_ref, x_ref, x)

    rho_point, p_point, _, _, _ = equilibrium.point_values(u0)

    assert np.all(np.abs(rho_point - rho_exact) < 1e-14)
