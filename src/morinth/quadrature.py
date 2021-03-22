import numpy as np

class GaussQuadrature:
    def __call__(self, edges, f):
        """Return the approximate cell-average of `f`."""
        qi = self.points
        dx = edges[1:] - edges[:-1]

        xi = 0.5*(qi + 1)*dx + edges[:-1]
        wi = 0.5*self.weights*dx

        return np.sum(wi*f(xi), axis=-1)/dx[:,0]

class GaussLegendre(GaussQuadrature):
    """Gauss-Legendre quadrature points and weights."""

    def __init__(self, n_points):
        if n_points == 1:
            self.points = np.array([0.0])
            self.weights = np.array([2.0])

        elif n_points == 2:
            sqrti3 = np.sqrt(1.0/3.0)
            self.points = np.array([-sqrti3, sqrti3])
            self.weights = np.array([1.0, 1.0])

        elif n_points == 3:
            sqrt35 = np.sqrt(3.0/5.0)
            self.points = np.array([-sqrt35, 0.0, sqrt35])
            self.weights = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])

        elif n_points == 4:
            inner = 2.0/7.0*np.sqrt(6.0/5.0)
            outer_minus = np.sqrt(3.0/7.0 - inner)
            outer_plus = np.sqrt(3.0/7.0 + inner)

            weight_plus = (18.0 - np.sqrt(30))/36.0
            weight_minus = (18.0 + np.sqrt(30))/36.0

            self.points = np.array([-outer_plus, -outer_minus, outer_minus, outer_plus])
            self.weights = np.array([weight_plus, weight_minus, weight_minus, weight_plus])

        elif n_points == 5:
            inner = 2.0*np.sqrt(10.0/7.0)
            outer_minus = 1.0/3.0*np.sqrt(5.0 - inner)
            outer_plus = 1.0/3.0*np.sqrt(5.0 + inner)

            weight_plus = (322.0 - 13.0*np.sqrt(70.0))/900.0
            weight_zero = 128.0/225.0
            weight_minus = (322.0 + 13.0*np.sqrt(70.0))/900.0

            self.points = np.array([-outer_plus, -outer_minus, 0.0, outer_minus, outer_plus])
            self.weights = np.array([weight_plus, weight_minus, weight_zero,
                                     weight_minus, weight_plus])

        else:
            raise Exception("Wrong `n_points`, try 1,...,6.")
