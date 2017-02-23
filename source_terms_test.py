import numpy as np

from source_terms import BalancedSourceTerm
from grid import Grid
from euler import Euler
from equilibrium import IsothermalEquilibrium
from weno import OptimalWENO, EquilibriumStencil
from gaussian_bump import GaussianBumpIC
from quadrature import GaussLegendre
from math_tools import linf_error, convergence_rate
from latex_tables import LatexConvergenceTable
from visualize import ConvergencePlot

def check_source_term_order(order):
    model = Euler(gamma=1.4, gravity = 1.2, specific_gas_constant=2.0)
    quadrature = GaussLegendre(5)
    all_resolutions = 2**np.arange(3, 10)

    ic = GaussianBumpIC(model)
    ic.p_amplitude, ic.rho_amplitude = 0.0, 10.0

    err = np.empty((4, all_resolutions.size))

    for l, res in enumerate(all_resolutions):
        grid = Grid([0.0, 1.0], res, 3)
        equilibrium = IsothermalEquilibrium(grid, model)
        weno = OptimalWENO(EquilibriumStencil(grid, equilibrium, model))
        source_term = BalancedSourceTerm(grid, model, equilibrium, order=order)

        u_bar = ic(grid)
        u_plus, u_minus = weno(u_bar, axis=0)
        u_left, u_right = u_minus[:,:-1,...], u_plus[:,1:,...]
        s_approx = source_term.edge_source(u_bar, u_left, u_right, axis=0)

        rho_bar = quadrature(grid.edges[3:-3,...], lambda x: ic.point_values(x)[0,...])
        v_bar = 0.0

        s_ref = np.zeros_like(s_approx)
        s_ref[1,...] = -rho_bar * model.gravity
        s_ref[3,...] = -v_bar * model.gravity

        err[:, l] = linf_error(s_approx, s_ref)

    rate = convergence_rate(err[1, ...], all_resolutions-6)

    return err[1,...], rate, all_resolutions


def test_source_term_order():
    all_errors, all_rates = [], []
    all_labels = ["$S_{(1)}$", "$S_{(2)}$"]

    for order in [2, 4]:
        errors, rates, resolutions = check_source_term_order(order)
        all_errors.append(errors)
        all_rates.append(rates)

    filename_base = "img/code-validation/source_term"
    table = LatexConvergenceTable(all_errors, all_rates, resolutions, all_labels)
    table.write(filename_base + ".tex")

    plot = ConvergencePlot(trend_orders=[2, 4])
    plot(all_errors, resolutions, all_labels)
    plot.save(filename_base)
