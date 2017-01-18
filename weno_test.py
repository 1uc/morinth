import numpy as np

from weno import WENO
from grid import Grid
from quadrature import GaussLegendre

import pytest

def sinusoidal(x):
    fx = np.sin(2.0*np.pi*x) * np.cos(2.0*np.pi*x)**2
    return fx

def weno_error(resolution):
    grid = Grid([0.0, 1.0], resolution, 3)
    x = grid.cell_centers

    cell_average = GaussLegendre(5)
    u0 = 1.0/grid.dx*cell_average(grid.edges, sinusoidal).reshape((1, -1))
    weno = WENO()

    u_plus, u_minus = weno(u0, axis=0)

    xij = grid.edges
    uref = sinusoidal(xij).reshape((1, -1))

    err_plus = np.max(np.abs(u_plus - uref[:,3:-2]))
    err_minus = np.max(np.abs(u_minus - uref[:,2:-3]))

    return err_plus, err_minus

def test_weno():
    resolutions = np.array([10, 20, 40, 80, 160]).reshape((-1,1))
    err = np.array([weno_error(int(res)) for res in list(resolutions)])

    rate = np.log(err[:-1,:]/err[1:,:]) / np.log(resolutions[:-1,:]/resolutions[1:,:])
    rate = np.max(np.abs(rate))
    assert np.abs(rate - 5.0) < 0.1, "Observed rate: {:.1e}".format(rate)

