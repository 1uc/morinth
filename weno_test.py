import numpy as np

from weno import OptimalWENO
from grid import Grid
from quadrature import GaussLegendre

import matplotlib.pylab as plt

import pytest

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
    uref = sinusoidal(xij).reshape((1, -1))

    err_plus = np.max(np.abs(u_plus - uref[:,3:-3]))
    err_minus = np.max(np.abs(u_minus - uref[:,3:-3]))

    return err_plus, err_minus

def test_weno_smooth():
    resolutions = np.array([16, 26, 46, 86, 166]).reshape((-1,1))
    err = np.array([weno_error(int(res)) for res in list(resolutions)])

    rate = np.log(err[:-1,:]/err[1:,:]) / np.log((resolutions[:-1,:] - 6)/(resolutions[1:,:] - 6))
    rate = np.max(np.abs(rate), axis=0)
    assert np.all(np.abs(rate - 5.0) < 0.1), "Observed rate: " + str(rate)

def increasing_criterium(u_plus, u_minus):
    assert np.all(u_plus <= u_minus + 1e-9)
    assert np.all(u_plus[:,1:] - u_plus[:,:-1] >= -1e-9)
    assert np.all(u_minus[:,1:] - u_minus[:,:-1] >= -1e-9)

def decreasing_criterium(u_plus, u_minus):
    assert np.all(u_plus >= u_minus - 1e-9)
    assert np.all(u_plus[:,1:] - u_plus[:,:-1] <= 1e-9)
    assert np.all(u_minus[:,1:] - u_minus[:,:-1] <= 1e-9)

@pytest.mark.skip(reason="This requires visual inspection.")
def test_weno_discontinuous():
    weno = OptimalWENO()
    resolutions = np.array([10, 20, 40]).reshape((-1,1))
    functions = [montone_increasing, montone_decreasing]
    criteria = [increasing_criterium, decreasing_criterium]

    for f, c in zip(functions, criteria):
        for resolution in resolutions:
            plt.clf()

            grid = Grid([0.0, 1.0], int(resolution), 3)

            cell_average = GaussLegendre(5)
            u0 = 1.0/grid.dx*cell_average(grid.edges, f).reshape((1, -1))

            u_plus, u_minus = weno(u0, axis=0)
            plt.plot(grid.edges[3:-3,0], u_plus[0,:], '>')
            plt.hold(True)
            plt.plot(grid.cell_centers[3:-3,0], u0[0,3:-3], 'k_')
            plt.plot(grid.edges[3:-3,0], u_minus[0,:], '<')
            plt.show()

            c(u_plus, u_minus)
