import numpy as np

from weno import OptimalWENO
from grid import Grid
from quadrature import GaussLegendre
from euler import Euler

import matplotlib.pylab as plt

import pytest
from testing_tools import is_manual_mode
from math_tools import convergence_rate, l1_error, linf_error

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

    assert_msg = lambda rate: "Observed rate: {:s}".format(str(rate))

    rate_minus = convergence_rate(err_minus, resolutions-6)
    print(err_minus)
    assert np.abs(np.max(rate_minus) - 5.0) < 0.1, assert_msg(rate_minus)

    rate_plus = convergence_rate(err_plus, resolutions-6)
    print(err_plus)
    assert np.abs(np.max(rate_plus) - 5.0) < 0.1, assert_msg(rate_plus)


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
    all_resolutions = np.array([10, 20, 40])
    functions = [montone_increasing, montone_decreasing]
    criteria = [increasing_criterium, decreasing_criterium]
    descriptions = ["increasing", "decreasing"]
    assert len(functions) == len(criteria) == len(descriptions)

    for f, c, d in zip(functions, criteria, descriptions):
        for resolution in all_resolutions:
            grid = Grid([0.0, 1.0], resolution, 3)

            cell_average = GaussLegendre(5)
            u0 = 1.0/grid.dx*cell_average(grid.edges, f).reshape((1, -1))

            u_plus, u_minus = weno(u0, axis=0)

            plt.clf()
            plt.plot(grid.edges[3:-3,0], u_plus[0,:], '>', label='from left')
            plt.plot(grid.cell_centers[3:-3,0], u0[0,3:-3], 'k_', label='average')
            plt.plot(grid.edges[3:-3,0], u_minus[0,:], '<', label='from right')
            plt.legend(loc="best")

            filename_pattern= "img/code-validation/weno_jump_{:s}-N{:d}"
            filename = filename_pattern.format(d, resolution)

            plt.savefig(filename + ".png")
            plt.savefig(filename + ".eps")

            if is_manual_mode():
                plt.show()

            plt.clf()

            c(u_plus, u_minus)
