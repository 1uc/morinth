# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

import numpy as np
import scipy.sparse.linalg as sparse_linalg

class ApproximateJacobian(object):
    def __init__(self, f, x, epsilon):
        self.epsilon = epsilon
        self.f = f
        self.x = x.reshape(-1)
        self.fx = f(x)

    def __call__(self, v):
        return (self.f(self.x + self.epsilon*v) - self.fx)/self.epsilon

    def evaluated_at(self, x):
        self.x = x.reshape(-1)
        self.fx = self.f(self.x)

    def as_operator(self):
        shape = (self.x.size, self.x.size)
        return sparse_linalg.LinearOperator(shape, matvec=self)
