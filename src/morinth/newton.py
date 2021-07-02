# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

import numpy as np
import scipy.sparse.linalg as sparse_linalg

class Newton(object):
    def __init__(self, boundary_mask):
        self.mask = boundary_mask.reshape(-1)

    def __call__(self, F, dF, x0):
        x, Fx = x0.reshape((-1)), F(x0.reshape((-1)))
        n_iter = 0
        while not self.is_converged(x, Fx):
            dF.evaluated_at(x)

            delta = np.zeros_like(x)
            delta, ret_code = sparse_linalg.gmres(dF.as_operator(), -Fx, delta, tol=1e-3)
            assert ret_code == 0, "ret_code = " + str(ret_code)

            x = x + delta
            Fx = F(x)
            n_iter += 1

        return x.reshape(x0.shape)

    def is_converged(self, x, Fx):
        return np.all(np.abs(Fx[self.mask]) < 1e-3)


