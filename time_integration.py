import numpy as np
from jacobian import approximate_jacobian
from newton import Newton

class TimeIntegration(object):
    """Interface for time-integration."""

    def __init__(self, bc, rate_of_change):
        self.bc = bc
        self.rate_of_change = rate_of_change

class ForwardEuler(TimeIntegration):
    """Simple forward Euler time-integration."""

    def __init__(self, bc, rate_of_change):
        super().__init__(bc, rate_of_change)
        self.cfl_number = 0.45

    def __call__(self, u0, t, dt):
        u1 = u0 + dt*self.rate_of_change(u0, t)
        self.bc(u1)

        return u1

    def pick_time_step(self, u):
        return self.cfl_number * self.rate_of_change.pick_time_step(u)

class BackwardEuler(TimeIntegration):
    def __init__(self, bc, rate_of_change):
        super().__init__(bc, rate_of_change)
        self.cfl_number = 10.0
        self.epsilon = 1e-8
        self.non_linear_solver = Newton()

    def __call__(self, u0, t, dt):
        F = self.NonlinearEquation(u0, t, dt, self.bc, self.rate_of_change)
        dF = approximate_jacobian(F, u0, self.epsilon)
        u1 = self.non_linear_solver(F, dF, u0)
        self.bc(u1)

        return u1

    def pick_time_step(self, u):
        return self.cfl_number * self.rate_of_change.pick_time_step(u)

    class NonlinearEquation(object):
        def __init__(self, u0, t, dt, bc, rate_of_change):
            self.u0, self.t, self.dt = u0, t, dt
            self.bc = bc
            self.rate_of_change = rate_of_change
            self.shape = u0.shape

        def __call__(self, u):
            u0, t, dt = self.u0, self.t, self.dt
            u = u.reshape(self.shape)

            self.bc(u)
            residual =  u - (u0 + dt*self.rate_of_change(u, t+dt))
            return residual.reshape((-1))

