# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

import numpy as np

from morinth.quadrature import GaussLegendre
from morinth.math_tools import gaussian, l1_error, l1rel_error, linf_error, convergence_rate

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
