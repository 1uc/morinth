# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

import numpy as np

from morinth.time_integration import ExplicitTimeIntegration
from morinth.butcher import ButcherTableau


class ExplicitRungeKutta(ExplicitTimeIntegration):
    """Base class for explicit Runge-Kutta methods.

    We write Runge-Kutta methods for du/dt = L(u, t) with s-stages as:
        u[j] = u0 + dt_j * sum_i a[j,i] * k[j]
        k[j] = L(u[j], t + c[j]*dt)

        u1 = u0 + dt * sum_j b[j]*k[j]

    """

    def __init__(self, bc, rate_of_change, tableau):
        super().__init__(bc, rate_of_change)
        self.tableau = tableau
        self.dudt_buffers = None

    def __call__(self, u0, t, dt):
        self.allocate_buffers(u0)

        k = self.dudt_buffers
        k[..., 0] = self.rate_of_change(u0, t)
        for s in range(1, self.tableau.stages):
            dt_star = self.tableau.c[s] * dt
            t_star = t + dt_star

            u_star = u0 + dt * np.sum(self.tableau.a[s, :s] * k[..., :s], axis=-1)
            self.bc(u_star, t_star)

            k[..., s] = self.rate_of_change(u_star, t_star)

        u1 = u0 + dt * np.sum(self.tableau.b * k, axis=-1)
        self.bc(u1, t + dt)

        return u1

    def __str__(self):
        return type(self).__name__ + "(..)"

    def __repr__(self):
        # fmt: off
        return "".join([type(self).__name__,
                        "(bc = ", repr(self.bc), ", ",
                        "rate_of_change = ", repr(self.rate_of_change), ", ",
                        "tableau = ", repr(self.tableau), ")"])
        # fmt: on

    def allocate_buffers(self, u):
        shape = u.shape + (self.tableau.stages,)

        if self.dudt_buffers is None or self.dudt_buffers.shape != shape:
            self.dudt_buffers = np.zeros(shape)


class ForwardEuler(ExplicitRungeKutta):
    """Simple forward Euler time-integration."""

    def __init__(self, bc, rate_of_change):
        a = np.array([[0.0]])
        b = np.array([1.0])

        super().__init__(bc, rate_of_change, ButcherTableau(a, b))
        self.cfl_number = 0.40


class SSP2(ExplicitRungeKutta):
    """Second order strong stability preserving RK method."""

    def __init__(self, bc, rate_of_change):
        # fmt: off
        a = np.array([[0.0, 0.0],
                      [1.0, 0.0]])
        b = np.array([0.5, 0.5])
        # fmt: on

        super().__init__(bc, rate_of_change, ButcherTableau(a, b))
        self.cfl_number = 0.45


class SSP3(ExplicitRungeKutta):
    """Third order strong stability preserving RK method."""

    def __init__(self, bc, rate_of_change):
        # fmt: off
        a = np.array([[ 0.0,  0.0,  0.0],
                      [ 1.0,  0.0,  0.0],
                      [ 0.25, 0.25, 0.0]])

        b = np.array([1.0/6, 1.0/6, 2.0/3])
        # fmt: on

        super().__init__(bc, rate_of_change, ButcherTableau(a, b))
        self.cfl_number = 0.40


class Fehlberg(ExplicitRungeKutta):
    """Fifth order scheme, not SSP"""

    def __init__(self, bc, rate_of_change):
        # fmt: off
        a = np.array([[ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [ 0.25, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [ 3.0/32.0, 9.0/32.0, 0.0, 0.0, 0.0, 0.0],
                      [ 1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0, 0.0, 0.0, 0.0],
                      [ 439.0/216.0, -8.0, 3680.0/513.0, -845.0/4104, 0.0, 0.0],
                      [ -8.0/27.0, 2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0, 0.0]])
        b = np.array([16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0])
        # fmt: on

        super().__init__(bc, rate_of_change, ButcherTableau(a, b))
        self.cfl_number = 0.3
