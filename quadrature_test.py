import numpy as np
import scipy.integrate

from quadrature import GaussLegendre
from grid import Grid

def sinusoidal(x):
    x = (x - 10.0)/100.0
    fx = np.sin(2.0*np.pi*x) * np.cos(2.0*np.pi*x)**2
    return fx

def quadrature_error(n_cells, n_points):
    grid = Grid([10.0, 100.0], n_cells, 1)
    integrate = GaussLegendre(n_points)

    fbar = 1.0/grid.dx*integrate(grid.edges, sinusoidal)
    fref = np.empty(n_cells)

    for i in range(n_cells):
        fref[i], err = scipy.integrate.quadrature(sinusoidal, grid.edges[i], grid.edges[i+1],
                                                  tol = 1e-10, rtol=1e-10)

    fref *= 1.0/grid.dx

    return np.max(np.abs(fbar - fref))

def quadrature_rate(n_points):
    resolutions = np.array([10, 20, 50, 100, 200, 500])
    errors = np.array([quadrature_error(int(n_cells), n_points) for n_cells in list(resolutions)])
    rate = np.max(np.abs(np.log(errors[1:]/errors[:-1])/np.log(resolutions[1:]/resolutions[:-1])))

    return rate

def test_quadrature_vs_scipy():
    for n_points in range(1,6):
        empirical = quadrature_rate(n_points)
        order = 2.0*n_points
        assert empirical > order - 0.1, "{:.1f} vs. {:.1f}".format(empirical, float(order))
