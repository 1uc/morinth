# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

import numpy as np

def fixpoint_iteration(f, x0, atol=1e-12, rtol=1e-12, maxiter=100, full_output=False):
    info = {'n_eval': 1}
    def is_converged(x0, fx0):
        delta = np.abs(fx0 - x0)
        return np.all(np.logical_or(delta < atol, delta/np.abs(x0) < rtol))

    x = x0
    fx = f(x)

    iter = 1
    while not is_converged(x, fx) and iter < maxiter:
        x, fx = fx, f(fx)
        info['n_eval'] += 1
        iter += 1

    if full_output:
        return fx, info
    else:
        return fx
