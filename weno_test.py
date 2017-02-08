import numpy as np

from weno import OptimalWENO, EquilibriumStencil
from grid import Grid
from quadrature import GaussLegendre
from euler import Euler
from equilibrium import IsothermalEquilibrium
from gaussian_bump import GaussianBumpIC

import matplotlib.pylab as plt

import pytest
from testing_tools import mark_manual
from math_tools import convergence_rate, linf_error

def sinusoidal(x):
    fx = np.sin(2.0*np.pi*x) * np.cos(2.0*np.pi*x)**2
    return fx

def montone_increasing(x):
    return np.sign(x - 0.51234567)

def montone_decreasing(x):
    return -np.sign(x - 0.51234567)

def weno_error(resolution):
    grid = Grid([0.0, 1.0], resolution, 3)
    x = grid.cell_centers

    cell_average = GaussLegendre(5)
    u0 = cell_average(grid.edges, sinusoidal).reshape((1, -1))
    weno = OptimalWENO()

    u_plus, u_minus = weno(u0, axis=0)

    xij = grid.edges
    u_ref = sinusoidal(xij).reshape((1, -1))

    err_plus = linf_error(u_plus, u_ref[:,3:-3])
    err_minus = linf_error(u_minus, u_ref[:,3:-3])

    return err_plus, err_minus

def test_weno_smooth():
    resolutions = np.array([16, 26, 46, 86, 166]).reshape((-1,1))
    err_plus = np.empty((1, resolutions.size))
    err_minus = np.empty_like(err_plus)

    for l, res in enumerate(np.nditer(resolutions)):
        err_plus[:,l], err_minus[:,l] = weno_error(res)

    rate_minus = convergence_rate(err_plus, resolutions-6)
    rate_plus = convergence_rate(err_minus, resolutions-6)
    rate = np.maximum(np.max(np.abs(rate_plus), axis=0),
                      np.max(np.abs(rate_minus), axis=0))
    assert np.abs(np.max(np.abs(rate)) - 5.0) < 0.1, "Observed rate: " + str(rate)

def increasing_criterium(u_plus, u_minus):
    assert np.all(u_plus <= u_minus + 1e-9)
    assert np.all(u_plus[:,1:] - u_plus[:,:-1] >= -1e-9)
    assert np.all(u_minus[:,1:] - u_minus[:,:-1] >= -1e-9)

def decreasing_criterium(u_plus, u_minus):
    assert np.all(u_plus >= u_minus - 1e-9)
    assert np.all(u_plus[:,1:] - u_plus[:,:-1] <= 1e-9)
    assert np.all(u_minus[:,1:] - u_minus[:,:-1] <= 1e-9)

def test_weno_discontinuous():
    weno = OptimalWENO()
    resolutions = np.array([10, 20, 40]).reshape((-1,1))
    functions = [montone_increasing, montone_decreasing]
    criteria = [increasing_criterium, decreasing_criterium]

    for f, c in zip(functions, criteria):
        for resolution in resolutions:
            plt.clf()

            grid = Grid([0.0, 1.0], resolution, 3)

            cell_average = GaussLegendre(5)
            u0 = 1.0/grid.dx*cell_average(grid.edges, f).reshape((1, -1))

            u_plus, u_minus = weno(u0, axis=0)
            # plt.plot(grid.edges[3:-3,0], u_plus[0,:], '>')
            # plt.hold(True)
            # plt.plot(grid.cell_centers[3:-3,0], u0[0,3:-3], 'k_')
            # plt.plot(grid.edges[3:-3,0], u_minus[0,:], '<')
            # plt.show()

            c(u_plus, u_minus)

@mark_manual
def test_weno_well_balanced():
    model = Euler(gamma = 1.4, gravity = 1.0, specific_gas_constant = 1.0)
    resolutions = np.array([16]).reshape((-1,1))
    p_ref, T_ref, x_ref = 1.0, 2.0, 0.0

    for resolution in resolutions:
        plt.clf()

        grid = Grid([0.0, 1.0], resolution, 3)
        equilibrium = IsothermalEquilibrium(grid, model)
        weno = OptimalWENO(EquilibriumStencil(grid, equilibrium, model))

        ic = GaussianBumpIC(model)
        ic.amplitude = 0.0
        u0 = ic(grid)
        u_plus, u_minus = weno(u0, axis=0)
        assert np.all(np.abs(u_plus - u_minus) < 1e-13)

        plt.plot(grid.edges[3:-3,0], u_plus[0,:], '>')
        plt.hold(True)
        plt.plot(grid.cell_centers[3:-3,0], u0[0,3:-3], 'k_')
        plt.plot(grid.edges[3:-3,0], u_minus[0,:], '<')
        plt.show()
