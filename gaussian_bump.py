#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
from matplotlib import rcParams
# matplotlib.rc('text', usetex = True)
rcParams.update({ 'font.family': 'sans-serif',
                  'font.size': 15,
                  'xtick.labelsize': 15,
                  'ytick.labelsize': 15,
                  'figure.autolayout': True,
                  'axes.formatter.limits': (-1, 3)})

import matplotlib.pyplot as plt
import pickle

from morinth.equilibrium import IsothermalEquilibrium, IsentropicEquilibrium
from morinth.euler_experiment import EulerExperiment
from morinth.boundary_conditions import Outflow, HydrostaticOutflow
from morinth.visualize import EquilibriumGraphs, DensityGraph, ConvergencePlot, DumpToDisk
from morinth.visualize import CombineIO, EulerGraphs, Markers
from morinth.weno import OptimalWENO, EquilibriumStencil
from morinth.source_terms import BalancedSourceTerm, UnbalancedSourceTerm
from morinth.math_tools import gaussian, l1_error, l1rel_error, linf_error, convergence_rate
from morinth.time_keeper import FixedSteps, PlotNever, PlotLast
from morinth.quadrature import GaussLegendre
from morinth.coding_tools import with_default
from morinth.latex_tables import LatexConvergenceTable
from morinth.euler import LinearGravity, PointMassGravity


class GaussianBumpIC(object):
    def __init__(self, model, equilibrium):
        self.model = model
        self.equilibrium = equilibrium

        self.p_amplitude = 1.0e-5
        self.rho_amplitude = 1.0e-5
        self.sigma = 0.05

        self.x_ref = 0.0
        self.rho_ref = 2.0
        self.p_ref = 1.0

        self.x_mid = 0.5
        self.p_mid = 0.3

        self.quadrature = GaussLegendre(5)

    @property
    def T_ref(self):
        return self.model.temperature(p=self.p_ref, rho=self.rho_ref)

    @property
    def u_ref(self):
        E_int_ref = self.model.internal_energy(p=self.p_ref)
        return np.array([self.rho_ref, 0.0, 0.0, E_int_ref])

    def __call__(self, grid):
        du_lambda = lambda x : self.delta(x, as_conserved_variables=True)
        u0 = self.back_ground(grid)
        du0 = self.quadrature(grid.edges, du_lambda)

        assert np.all(np.isfinite(u0))
        assert np.all(np.isfinite(du0))

        return u0 + du0

    def delta(self, x, as_conserved_variables=False):
        """Perturbation in respective variables."""

        dp = self.p_mid*gaussian(x - self.x_mid, self.sigma)
        drho = self.model.rho(p=dp, T=self.T_ref)

        dw = np.zeros((4,) + dp.shape)
        dw[0,...] = self.rho_amplitude * drho
        dw[3,...] = self.p_amplitude * dp

        if as_conserved_variables:
            dw[3,...] = self.model.internal_energy(p=dw[3,...])

        return dw

    def back_ground(self, grid):
        x = grid.cell_centers[:,0]
        equilibrium = self.equilibrium
        u_bar_ref = equilibrium.cell_averages(self.u_ref, self.x_ref)

        return equilibrium.reconstruct(u_bar_ref, self.x_ref, x)

    def point_values(self, x):
        """Point values of the primitive variables."""

        equilibrium = self.equilibrium
        rho_ref, p_ref, x_ref = self.rho_ref, self.p_ref, self.x_ref

        w = np.zeros((4,) + x.shape)
        w[0,...], w[3,...] = equilibrium.extrapolate(rho_ref, p_ref, x_ref, x)
        dw = self.delta(x)

        return w + dw

class ExtremeGaussianBumpIC(GaussianBumpIC):
    def __init__(self, model, equilibrium):
        self.model = model
        self.equilibrium = equilibrium

        self.p_amplitude = 1.0e-6
        self.rho_amplitude = 1.0e-6

        self.sigma = 0.05*(self.domain[1] - self.domain[0])

        self.x_ref = 0.0
        self.rho_ref = 32.0
        self.p_ref = 10000.0

        self.x_mid = 0.5*(self.domain[0] + self.domain[1])
        self.p_mid = 1000.0

        self.quadrature = GaussLegendre(5)

    @property
    def domain(self):
        return np.array([0.0, 500.0])


class EquilibriumExperiment(EulerExperiment):
    @property
    def gravity(self):
        return 1.0

    @property
    def n_cells(self):
        return self._n_cells

    @n_cells.setter
    def n_cells(self, rhs):
        self._n_cells = rhs

    @property
    def order(self):
        return 5

    @property
    def weno(self):
        mode = self.well_balancing

        if mode == "wb_o2" or mode == "wb_o4":
            return OptimalWENO(EquilibriumStencil(self.grid, self.equilibrium, self.model))

        elif mode == "naive":
            return OptimalWENO()

        else:
            raise Exception("Wrong `well_balancing`.")

    @property
    def thermodynamic_equilibrium(self):
        label = self._thermodynamic_equilibrium
        if label == "isentropic":
            return IsentropicEquilibrium
        elif label == "isothermal":
            return IsothermalEquilibrium
        else:
            raise Exception("Unknown equilibrium.")

    @thermodynamic_equilibrium.setter
    def thermodynamic_equilibrium(self, rhs):
        self._thermodynamic_equilibrium = rhs

    @property
    def boundary_condition(self):
        equilibrium = self.thermodynamic_equilibrium(self.grid, self.model)
        return HydrostaticOutflow(self.grid, equilibrium)

    @property
    def source(self):
        mode = self.well_balancing

        if mode == "wb_o2" or mode == "wb_o4":
            source_order = 4.0 if mode == "wb_o4" else 2.0
            return BalancedSourceTerm(self.grid, self.model,
                                      self.equilibrium, source_order)

        elif mode == "naive":
            return UnbalancedSourceTerm(self.grid, self.model)

        else:
            raise Exception("Wrong `well_balancing`.")

    @property
    def well_balancing(self):
        return self._well_balancing

    @well_balancing.setter
    def well_balancing(self, rhs):
        self._well_balancing = rhs

    @property
    def equilibrium(self):
        mode = self.well_balancing
        if mode == "wb_o2" or mode == "wb_o4":
            return self.thermodynamic_equilibrium(self.grid, self.model)

        elif mode == "naive":
            return None

        else:
            assert Exception("Wrong `well_balancing`.")

    @property
    def visualize(self):
        back_ground = self.initial_condition.back_ground(self.grid)
        graphs = EquilibriumGraphs(self.grid,
                                   "img/" + self.output_filename,
                                   self.model,
                                   back_ground)
        raw_data = DumpToDisk(self.grid, "data/" + self.output_filename, back_ground)

        return CombineIO(graphs, raw_data)

    @property
    def output_filename(self):
        pattern = self.base_filename + "-{:s}_{:s}_res{:05d}"
        return pattern.format(self._thermodynamic_equilibrium,
                              self.well_balancing,
                              self.n_cells)


class GaussianBump(EquilibriumExperiment):
    @property
    def final_time(self):
        return 8.0

    @property
    def specific_gas_constant(self):
        return 0.01

    @property
    def domain(self):
        return np.array([0, 500])

    @property
    def initial_condition(self):
        equilibrium = self.thermodynamic_equilibrium(self.grid, self.model)
        ic = ExtremeGaussianBumpIC(self.model, equilibrium)
        ic.p_amplitude, ic.rho_amplitude = 1e-6, 1e-6

        return ic

    @property
    def gravity(self):
        return PointMassGravity(gravitational_constant=800.0,
                                mass=3000.0,
                                radius=1000.0)

    @property
    def base_filename(self):
        return  "extreme_gaussian_bump"

    @property
    def steps_per_frame(self):
        return 5

class GaussianBumpConvergence(GaussianBump):
    @property
    def plotting_steps(self):
        # return super().plotting_steps
        return PlotLast()


class GaussianBumpReference(GaussianBumpConvergence):
    @property
    def well_balancing(self):
        return "naive"

    @property
    def n_cells(self):
        return 2**14 + 6

    @property
    def output_filename(self):
        pattern = "extreme_gaussian_bump-{:s}"
        return pattern.format(self._thermodynamic_equilibrium)


def compute_reference_solution(Experiment, thermodynamic_equilibrium):
    experiment = Experiment()
    experiment.thermodynamic_equilibrium = thermodynamic_equilibrium

    grid = experiment.grid

    u0 = experiment.initial_condition.back_ground(grid)
    u_ref = experiment()

    filename_base = "data/" + experiment.output_filename
    np.save(filename_base + "_background.npy", u0)
    np.save(filename_base + "_reference.npy", u_ref)

    with open(filename_base + "_grid.pkl", 'wb') as f:
        pickle.dump(grid, f)


def load_reference_solution(Experiment, thermodynamic_equilibrium):
    experiment = Experiment()
    experiment.thermodynamic_equilibrium = thermodynamic_equilibrium

    filename_base = "data/" + experiment.output_filename
    u0_ref = np.load(filename_base + "_background.npy")
    u_ref = np.load(filename_base + "_reference.npy")

    with open(filename_base + "_grid.pkl", 'rb') as f:
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


class EquilibriumConvergenceRates:
    def __call__(self, Experiment, ExperimentReference, thermal_equilibrium):
        all_errors, all_rates = [], []
        all_labels = self.all_labels

        if self.is_reference_solution_required:
            compute_reference_solution(ExperimentReference, thermal_equilibrium)

        for well_balancing in ["naive", "wb_o2", "wb_o4"]:
        # for well_balancing in ["naive", "wb_o4"]:
        # for well_balancing in ["wb_o2"]:
        # for well_balancing in ["wb_o4"]:
            error, rate, resolutions = self.compute_convergence(Experiment,
                                                        ExperimentReference,
                                                        thermal_equilibrium,
                                                        well_balancing)
            all_errors += error
            all_rates += rate

        experiment = ExperimentReference()
        experiment.thermodynamic_equilibrium = thermal_equilibrium
        filename_base = "".join(["img/code-validation/",
                                 experiment.base_filename,
                                 "-{:s}".format(experiment._thermodynamic_equilibrium)])
        latex_table = LatexConvergenceTable(all_errors,
                                            all_rates,
                                            resolutions-6,
                                            all_labels)
        latex_table.write(filename_base + ".tex")

        plot = ConvergencePlot(self.trend_lines)
        plot(all_errors, resolutions-6, all_labels)
        plot.save(filename_base)

    def compute_convergence(self, Experiment,
                                  ExperimentReference,
                                  thermal_equilibrium,
                                  well_balancing):

        u0_ref, u_ref, grid_ref = load_reference_solution(ExperimentReference,
                                                          thermal_equilibrium)
        du_ref = u_ref - u0_ref

        # plt.clf()
        # marker = iter(Markers())
        # self.plot_delta(grid_ref, u_ref, du_ref, None, next(marker))

        resolutions = self.resolutions

        err = np.empty((4, resolutions.size))
        for l, res in enumerate(resolutions):
            experiment = Experiment()
            experiment.thermodynamic_equilibrium = thermal_equilibrium
            experiment.well_balancing = well_balancing
            experiment.n_cells = res
            grid = experiment.grid
            n_ghost = grid.n_ghost

            u0 = experiment.initial_condition.back_ground(grid)

            u = experiment()

            du = u - u0
            du_ref_c = down_sample(du_ref, grid_ref, grid)

            # self.plot_delta(grid, u, du, du_ref_c, next(marker))
            err[:,l] = l1rel_error(du[:,n_ghost:-n_ghost], du_ref_c, ref=u_ref)

        # plt.show()
        error_vars = self.error_vars
        rates = [convergence_rate(err[k,...], resolutions-6) for k in error_vars]
        errors = [err[k,...] for k in error_vars]

        return errors, rates, resolutions

    def plot_delta(self, grid, u, du, du_ref_c, marker):
        n_ghost = grid.n_ghost
        x = grid.cell_centers[:,0]
        plt.plot(x, du[0,...], marker = marker)


class GaussianBumpConvergenceRates(EquilibriumConvergenceRates):
    def __init__(self):
        super().__init__()

        self.all_labels = ["$\\rho_{(0)}$", "$E_{(0)}$",
                           "$\\rho_{(1)}$", "$E_{(1)}$",
                           "$\\rho_{(2)}$", "$E_{(2)}$"]

        self.error_vars = [0, 3]
        self.resolutions = 2**np.arange(4, 11) + 6
        self.is_reference_solution_required = True
        self.trend_lines = [5]

if __name__ == "__main__":
    sim = GaussianBumpConvergenceRates()

    # sim(GaussianBumpConvergence, GaussianBumpReference, "isothermal")
    sim(GaussianBumpConvergence, GaussianBumpReference, "isentropic")
