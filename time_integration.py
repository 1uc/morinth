import numpy as np
from jacobian import ApproximateJacobian
from newton import Newton

class TimeIntegration(object):
    """Interface for time-integration."""

    def __init__(self, bc, rate_of_change):
        self.bc = bc
        self.rate_of_change = rate_of_change

class ExplicitTimeIntegration(TimeIntegration):
    """Base class for explicit solvers."""

    def pick_time_step(self, u):
        return self.cfl_number * self.rate_of_change.pick_time_step(u)

class ImplicitTimeIntegration(TimeIntegration):
    """Base class for implicit time integration."""

    def __init__(self, bc, rate_of_change, boundary_mask):
        super().__init__(bc, rate_of_change)
        self.epsilon = 1e-8
        self.non_linear_solver = Newton(boundary_mask)

class BackwardEuler(ImplicitTimeIntegration):
    def __init__(self, bc, rate_of_change, boundary_mask):
        super().__init__(bc, rate_of_change, boundary_mask)
        self.cfl_number = 3.0

    def __call__(self, u0, t, dt):
        F = self.NonlinearEquation(u0, t, dt, self.bc, self.rate_of_change)
        dF = ApproximateJacobian(F, u0, self.epsilon)
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

            # self.bc(u)
            residual =  u - (u0 + dt*self.rate_of_change(u, t+dt))
            return residual.reshape((-1))

class BDF2(ImplicitTimeIntegration):
    """Backward differentiation formula with two steps.

    The BDF2 is
        u2 - 4/3 u1 + 1/3 u0 = 2/3 h f(t2, u2).
    """

    def __init__(self, bc, rate_of_change, boundary_mask, fixed_dt):
        super().__init__(bc, rate_of_change, boundary_mask)
        self.backward_euler = BackwardEuler(bc, rate_of_change, boundary_mask)

        self.u0 = None
        self.fixed_dt = fixed_dt

    def __call__(self, u1, t, dt):
        if not self.u0:
            # The update from t=0 to t=dt must be done by BDF1.
            self.u0 = u1
            return self.backward_euler(self.u0, t, dt)

        F = self.NonlinearEquation(self.u0, u1, t, dt, self.bc, self.rate_of_change)
        dF = ApproximateJacobian(F, u1, self.epsilon)
        u2 = self.non_linear_solver(F, dF, u1)
        self.bc(u2)

        # store `u1` for use as `u0` in next iteration.
        self.u0 = u1

        return u2

    def pick_time_step(self, u):
        return self.fixed_dt

    class NonlinearEquation(object):
        def __init__(self, u0, u1, t, dt, bc, rate_of_change):
            self.u0, self.u1, self.t, self.dt = u0, u1, t, dt
            self.bc = bc
            self.rate_of_change = rate_of_change
            self.shape = u0.shape

        def __call__(self, u):
            u0, u1, t, dt = self.u0, self.u1, self.t, self.dt
            u = u.reshape(self.shape)

            self.bc(u)
            residual =  u - 4/3*u1 + 1/3*u0 - 2/3*dt*self.rate_of_change(u, t+dt)
            return residual.reshape((-1))
