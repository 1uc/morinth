import numpy as np
import scipy.sparse.linalg as sparse_linalg

class ApproximateJacobian(object):
    def __init__(self, f, x, epsilon):
        self.epsilon = epsilon
        self.f = f
        self.x = x
        self.fx = f(x)

    def __call__(self, v):
        return (self.f(self.x + self.epsilon*v) - self.fx)/self.epsilon

def approximate_jacobian(f, x, epsilon):
    jacobian = ApproximateJacobian(f, x, epsilon)
    shape = (x.size, x.size)
    return sparse_linalg.LinearOperator(shape, matvec=jacobian)
