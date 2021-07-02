# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

import numpy as np

from morinth.jacobian import ApproximateJacobian

class MockODE(object):
    def __init__(self):
        self.epsilon = 1e-8
        self.x = np.array([1.0, 2.0])
        self.f = lambda x : np.array([x[0]*x[1], x[0]**2])
        self.jac = lambda x: np.array([[x[1], x[0]], [2*x[0], 0]])


def test_fintie_difference():
    ode = MockODE()
    epsilon, f, jac, x = ode.epsilon, ode.f, ode.jac, ode.x
    ajac = ApproximateJacobian(f, x, epsilon)
    ojac = ajac.as_operator()

    v = np.random.random((2,))
    dfdv = np.matmul(jac(x),v)

    assert np.all(np.abs(dfdv - ajac(v)) < 5.0*epsilon)
    assert np.all(np.abs(dfdv - ojac(v)) < 5.0*epsilon)
