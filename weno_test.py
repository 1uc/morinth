import numpy as np

from weno import OptimalWENO, LagrangePolynomials
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

def weno_error(resolution, x_rel):
    grid = Grid([0.0, 1.0], resolution, 3)
    x = grid.cell_centers[2:-2,...,0] + x_rel*grid.dx

    cell_average = GaussLegendre(5)
    u0 = cell_average(grid.edges, sinusoidal).reshape((1, -1))
    weno = OptimalWENO()

    u_approx = weno.trace_values(u0, x_rel=x_rel)
    u_ref = sinusoidal(x).reshape((1, -1))

    return linf_error(u_approx, u_ref)

def weno_convergence(x_rel):
    resolutions = np.array([10, 20, 40, 80, 160, 320, 640]) + 6
    err = np.empty((1, resolutions.size))

    for l, res in enumerate(resolutions):
        err[:,l] = weno_error(res, x_rel)

    assert_msg = lambda rate: "Observed rate: {:s}".format(str(rate))

    rate = convergence_rate(err, resolutions-6)
    expected_rate = 4.0 if x_rel == 0.0 else 5.0
    assert np.all(np.abs(rate[:,-4:] - expected_rate) < 0.1), assert_msg(rate)

def test_weno_smooth():
    x_rel = [-0.5, -0.25, 0.0, 0.25, 0.5]
    for x in x_rel:
        weno_convergence(x)

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

def test_lagrange_prime():
    L = LagrangePolynomials()

    x = 0.23456
    xi = np.arange(6) + 0.5

    L1 = L.prime(x, xi[:4], 1)
    assert L1 == 0.5*(  (x - xi[2])*(x - xi[3])
                      + (x - xi[0])*(x - xi[3])
                      + (x - xi[0])*(x - xi[2]))

def test_p123():
    p_ref = np.empty((3, 3))
    p_ref[0,:] = np.array([1/3, -7/6, 11/6])
    p_ref[1,:] = np.array([-1/6, 5/6, 1/3])
    p_ref[2,:] = np.array([1/3, 5/6, -1/6])

    weno = OptimalWENO()
    p, _ = weno.compute_pq(x_rel=0.5)

    for i in range(3):
        assert np.all(np.abs(p[i,:] - p_ref[i,:]) < 1e-10), "i = {:d}".format(i)

def test_linear_weights():
    C_ref_05 = np.array([1.0/10.0, 3.0/5.0, 3.0/10.0])
    # C_ref_00 = np.array([-0.1125, 1.225, -0.1125])
    C_ref_00 = np.array([1, 100, 1])

    weno = OptimalWENO()

    for C_ref, x in zip([C_ref_05, C_ref_00], [0.5, 0.0]):
        C = np.array(weno.linear_weights(x))
        assert np.all(np.abs(C - C_ref) < 1e-12)

    assert 0.5 in weno._linear_weights
    assert 0.0 in weno._linear_weights
