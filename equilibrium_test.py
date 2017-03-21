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

        ic_point_values = lambda x: model.conserved_variables(ic.point_values(x))
        reference = quadrature(grid.edges, ic_point_values)

        err[:,l] = l1_error(ic(grid), reference)

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
    resolutions = 2**np.arange(4, 11) + 6
    all_x_rel = [-0.5, -0.25, 0.0, 0.25, 0.5]

    ic = GaussianBumpIC(model)
    ic.p_amplitude, ic.rho_amplitude = 1e-5, 1e-0

    err_rho = np.empty((5, len(resolutions)))
    rate_rho = np.empty((5, len(resolutions)-1))

    for k, x_rel in enumerate(all_x_rel):
        for l, resolution in enumerate(resolutions):
            grid = Grid([0.0, 1.0], resolution, 3)
            equilibrium = IsothermalEquilibrium(grid, model)
            weno = OptimalWENO(EquilibriumStencil(grid, equilibrium, model))

            u0 = ic(grid)
            u_weno = weno.trace_values(u0, x_rel=x_rel)[:,1:-1,...]

            x = grid.cell_centers[3:-3,...,0] + x_rel*grid.dx
            u_exact = ic.point_values(x)

            err_rho[k,l] = linf_error(u_weno[0,...], u_exact[0,...])

        rate_rho[k,:] = convergence_rate(err_rho[k,...], resolutions-6)

    all_errors = [err_rho[k,...] for k in range(err_rho.shape[0])]
    all_rates = [rate_rho[k,...] for k in range(rate_rho.shape[0])]
    all_labels = ["$\\rho({:.2f})$".format(x_rel) for x_rel in all_x_rel]

    filename_base = "img/code-validation/weno_well-balanced"

    table = LatexConvergenceTable(all_errors, all_rates, resolutions-6, all_labels)
    table.write(filename_base + ".tex")

    plot = ConvergencePlot(trend_orders=[5])
    plot(all_errors, resolutions-6, all_labels)
    plot.save(filename_base)

    def achieves(observed_rate, expected):
        assert np.all(np.abs(observed_rate[-2:] - expected) < 0.1)

    for i, x_rel in enumerate(all_x_rel):
        expected_rate = 5.0 if x_rel != 0.0 else 4.0
        assert np.all(np.abs(rate_rho[i,-3:] - expected_rate) < 0.2)


def test_cell_averages_to_points_values():
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

    rho_point, p_point, _ = equilibrium.point_values(u0, x)

    assert np.all(np.abs(rho_point - rho_exact) < 1e-14)
