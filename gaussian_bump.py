#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pickle

from equilibrium import IsothermalEquilibrium
from euler_experiment import EulerExperiment
from boundary_conditions import Outflow, IsothermalOutflow
from visualize import EquilibriumGraphs, DensityGraph
from weno import OptimalWENO, EquilibriumStencil
from source_terms import BalancedSourceTerm
from math_tools import gaussian, l1_error, convergence_rate
from time_keeper import FixedSteps, PlotNever, PlotLast
from quadrature import GaussLegendre
from coding_tools import with_default

class GaussianBumpIC(object):
    def __init__(self, model):
        self.model = model
        self.amplitude = 1.0e-5
        self.sigma = 0.05
        self.x_ref = 0.0
        self.x_mid = 0.5
        self.rho_ref = 1.0
        self.p_ref = 1.0
        E_int_ref = model.internal_energy(p=self.p_ref)
        self.u_ref = np.array([self.rho_ref, 0.0, 0.0, E_int_ref])

    def __call__(self, grid):
        x = grid.cell_centers[:,0]

        u0 = self.back_ground(grid)
        alpha = 1.0 + self.amplitude*gaussian(x - self.x_mid, self.sigma)
        # u0[0,...] = u0[0,...]/alpha
        u0[3,...] = u0[3,...]*alpha

        return u0

    def back_ground(self, grid):
        x = grid.cell_centers[:,0]
        equilibrium = IsothermalEquilibrium(model=self.model, grid=grid)
        u_bar_ref = equilibrium.cell_averages(self.u_ref)

        return equilibrium.reconstruct(u_bar_ref, self.x_ref, x)


class GaussianBump(EulerExperiment):
    @property
    def final_time(self):
        return 0.2

    @property
    def n_cells(self):
        return getattr(self, "_n_cells", 200)

    @n_cells.setter
    def n_cells(self, new_value):
        self._n_cells = new_value

    @property
    def initial_condition(self):
        return GaussianBumpIC(self.model)

    @property
    def order(self):
        return 5

    @property
    def gravity(self):
        return 1.0

    @property
    def boundary_condition(self):
        return IsothermalOutflow(self.grid, self.model)

    @property
    def weno(self):
        return OptimalWENO(EquilibriumStencil(self.grid, self.equilibrium, self.model))

    @property
    def source(self):
        return BalancedSourceTerm(self.grid, self.model, self.equilibrium, self.source_order)

    @property
    def source_order(self):
        return 2

    @property
    def equilibrium(self):
        return IsothermalEquilibrium(self.grid, self.model)

    @property
    def visualize(self):
        back_ground = self.initial_condition.back_ground(self.grid)
        return EquilibriumGraphs(self.grid, self.output_filename, self.model, back_ground)

    @property
    def output_filename(self):
        return "img/gaussian_bump"

    @property
    def steps_per_frame(self):
        return 1

class GaussianBumpConvergence(GaussianBump):
    @property
    def plotting_steps(self):
        return PlotLast()

    @property
    def output_filename(self):
        return "img/gaussian_bump_{:05d}".format(int(self.n_cells))

class GaussianBumpReference(GaussianBumpConvergence):
    @property
    def n_cells(self):
        return 2**12 + 6

    @property
    def source_order(self):
        return 2

def compute_reference_solution():
    gaussian_bump = GaussianBumpReference()
    grid = gaussian_bump.grid

    u0 = gaussian_bump.initial_condition.back_ground(grid)
    u_ref = gaussian_bump()

    np.save("data/gaussian_bump_background.npy", u0)
    np.save("data/gaussian_bump_reference.npy", u_ref)

    with open("data/gaussian_bump_grid.pkl", 'wb') as f:
        pickle.dump(grid, f)

    return u0, u_ref, grid

def load_reference_solution():
    u0_ref = np.load("data/gaussian_bump_background.npy")
    u_ref = np.load("data/gaussian_bump_reference.npy")

    with open("data/gaussian_bump_grid.pkl", 'rb') as f:
        grid = pickle.load(f)

    return u0_ref, u_ref, grid

def down_sample(u_fine, grid_fine, grid_coarse):
    """Compute cell-averages of `u_fine` on the coarse grid."""
    if grid_fine.n_dims == 2:
        raise Exception("Needs to be implemented.")

    ngf = grid_fine.n_ghost
    ngc = grid_coarse.n_ghost

    ncf = grid_fine.n_cells[0] - 2*ngf
    ncc = grid_coarse.n_cells[0] - 2*ngc
    r = ncf // ncc
    assert r*ncc == ncf

    shape = (u_fine.shape[0], -1, r)
    u_coarse = np.mean(u_fine[:,ngf:-ngf].reshape(shape), axis=-1)

    return u_coarse

if __name__ == '__main__':
    # u0_ref, u_ref, grid_ref = compute_reference_solution()
    u0_ref, u_ref, grid_ref = load_reference_solution()
    du_ref = u_ref - u0_ref

    all_resolutions = 2**np.arange(4, 10) + 6

    err = np.empty((4, all_resolutions.size))
    for l, res in enumerate(np.nditer(all_resolutions)):
        gaussian_bump = GaussianBumpConvergence()
        gaussian_bump.n_cells = res
        grid = gaussian_bump.grid
        n_ghost = grid.n_ghost

        u0 = gaussian_bump.initial_condition.back_ground(grid)
        u = gaussian_bump()

        du = u - u0

        ic = gaussian_bump.initial_condition
        du_ref_c = down_sample(du_ref, grid_ref, grid)
        err[:,l] = l1_error(du[:,n_ghost:-n_ghost], du_ref_c)

    rho_rate = convergence_rate(err[0,...], all_resolutions-6)
    E_int_rate = convergence_rate(err[3,...], all_resolutions-6)

    print(rho_rate)
    print(E_int_rate)
