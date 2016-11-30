import numpy as np
from time_integration import ExplicitTimeIntegration

class ButcherTableau(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.c = np.sum(a, axis=1)
        self.stages = a.shape[0]

    def is_explicit(self):
        """Does the tableau describe and explicit RK scheme?"""
        for i in range(self.stages):
            if not np.all(self.a[i,i:] == np.zeros((1, self.stages - i))):
                return False

        return True

    def is_implicit(self):
        return not self.is_explicit()

class ExplicitRungeKutta(ExplicitTimeIntegration):
    """Base class for explicit Runge-Kutta methods."""

    def __init__(self, bc, rate_of_change, tableau, shape):
        super().__init__(bc, rate_of_change)
        self.tableau = tableau
        self.dudt_buffers = np.zeros(shape+(tableau.stages,))

    def __call__(self, u0, t, dt):
        k = self.dudt_buffers

        k[...,0] = self.rate_of_change(u0, t)
        for s in range(1, self.tableau.stages):
            dt_star = tableau.c[s]*dt
            t_star = t + dt_star

            u_star = u0 + dt_star*np.sum(tableau.a[s,:s]*k[...,:s], axis=-1)
            self.bc(u_star)

            k[...,s] = self.rate_of_change(u_star, t_star)

        u1 = u0 + dt*np.sum(tableau.b*k, axis=-1)
        self.bc(u1)

        return u1

class ForwardEuler(ExplicitRungeKutta):
    """Simple forward Euler time-integration."""

    def __init__(self, bc, rate_of_change):
        a = np.array([[0.0]])
        b = np.array([1.0])

        super().__init__(bc, rate_of_change, ButcherTableau(a, b))
        self.cfl_number = 0.45


class SSP2(ExplicitRungeKutta):
    """Second order strong stability preserving RK method."""

    def __init__(self, bc, rate_of_change):
        a = np.array([[0.0, 0.0],
                      [1.0, 0.0]])
        b = np.array([0.5, 0.5])

        super().__init__(bc, rate_of_change, ButcherTableau(a, b))
        self.cfl_number = 0.85

class SSP3(ExplicitRungeKutta):
    """Third order strong stability preserving RK method."""

    def __init__(self, bc, rate_of_change):
        a = np.array([[ 0.0,  0.0,  0.0],
                      [ 1.0,  0.0,  0.0],
                      [ 0.25, 0.25, 0.0]])

        b = np.array([1.0/6, 1.0/6, 2.0/3])

        super().__init__(bc, rate_of_change, ButcherTableau(a, b))
        self.cfl_number = 0.85
