import numpy as np

from euler import Euler, QuasiLinearEuler

def test_eigenvector():
    model = QuasiLinearEuler(gravity = 0.0, gamma = 1.4, specific_gas_constant = 1.0)
    w = np.array([1.302, 1.0324, 0.0, 3.092])

    for axis in range(2):
        if axis == 1:
            w[2] = 1.2234

        eigenvalues = model.eigenvalues(w, axis=axis)
        eigenvectors = model.eigenvectors(w, axis=axis)
        matrix = model.coefficient_matrix(w, axis=axis)

        for k in range(4):
            Av = np.matmul(matrix, eigenvectors[:,k])
            lambda_v = eigenvalues[k] * eigenvectors[:,k]

            assert np.all(np.abs(Av - lambda_v) < 1e-9), str(k)
