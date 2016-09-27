import numpy as np
import scipy.sparse.linalg as sparse_linalg

class Newton(object):
    def __call__(self, F, dF, x0):
        x, Fx = x0.reshape((-1)), F(x0.reshape((-1)))
        while not self.is_converged(x, Fx):
            delta = np.zeros_like(x)
            delta, ret_code = sparse_linalg.gmres(dF, -Fx, delta)
            assert ret_code == 0

            x = x + delta
            Fx = F(x)

        return x.reshape(x0.shape)

    def is_converged(self, x, Fx):
        return np.all(np.linalg.norm(Fx) < 1e-8)


