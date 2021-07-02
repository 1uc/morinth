# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

import numpy as np

class ShallowWater(object):
    def __init__(self, gravity):
        self.gravity = gravity

    def flux(self, u, axis):
        f = np.empty_like(u)

        if axis == 0:
            v = self.velocity(u, axis)

            f[0,:,...] = u[axis+1,:,...]
            f[1,:,...] = u[1,:,...]*v + 0.5*self.gravity*u[0,...]*u[0,...]

        else:
            raise Exception("Not implemented.")

        return f

    def source(self, u):
        return 0.0

    def max_eigenvalue(self, u):
        return np.abs(u[1,...]/u[0,...]) + np.sqrt(self.gravity*u[0,...])

    def velocity(self, u, axis):
        return u[axis+1,...]/u[0,...]
