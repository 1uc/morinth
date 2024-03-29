# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

import numpy as np


class Burgers(object):
    """Burgers equation is a simple non-linear PDE."""

    def flux(self, u, axis):
        return 0.5 * u[axis, ...] * u

    def source(self, u, x):
        return 0.0

    def max_eigenvalue(self, u):
        return np.abs(u)


class VariableBurgers(object):
    """Burgers equation is a simple non-linear PDE with a coefficient."""

    def __init__(self, a):
        self.a = a

    def flux(self, u, axis):
        return 0.5 * self.a * u[axis, ...] * u

    def source(self, u, x):
        return 0.0

    def max_eigenvalue(self, u):
        return np.abs(u)
