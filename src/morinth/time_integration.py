# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

import numpy as np

from morinth.jacobian import ApproximateJacobian
from morinth.newton import Newton
from morinth.butcher import ButcherTableau

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

    def __str__(self):
        return type(self).__name__ + "(..)"

    def __repr__(self):
        return "".join([type(self).__name__,
                        "(bc = ", repr(self.bc), ", ",
                        "rate_of_change = ", repr(self.rate_of_change), ", ",
                        "boundary_mask = ", repr(self.boundary_mask), ")"])


class BackwardEuler(ImplicitTimeIntegration):
    def __init__(self, bc, rate_of_change, boundary_mask, cfl_number):
        super().__init__(bc, rate_of_change, boundary_mask)
        self.cfl_number = cfl_number

    def __call__(self, u0, t, dt):
        F = self.NonlinearEquation(u0, t, dt, self.bc, self.rate_of_change)
        dF = ApproximateJacobian(F, u0, self.epsilon)
        u1 = self.non_linear_solver(F, dF, u0)
        self.bc(u1, t+dt)

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
        self.backward_euler = BackwardEuler(bc, rate_of_change, boundary_mask, None)

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
        self.bc(u2, t + dt)

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

            self.bc(u, t + dt)
            residual =  u - 4/3*u1 + 1/3*u0 - 2/3*dt*self.rate_of_change(u, t+dt)
            return residual.reshape((-1))

class DIRK(ImplicitTimeIntegration):
    """Diagonally implicit Runge-Kutta (DIRK) schemes.

    DIRKs are implicit Runge-Kutta schemes where the only implicit terms
    are on the diagonal.

    References:
        [1]: Roger Alexander, 1977, SIAM J. Num. Anal.
    """

    def __init__(self, bc, rate_of_change, boundary_mask, cfl_number, tableau):
        """Create a DIRK object.

        Parameters:
            :bc: boundary conditions
            :rate_of_change: right-hand side of the ODE
            :boundary_mask: like in `ForwardEuler`
            :tableau: a diagonally implicit Butcher tableau
        """
        super().__init__(bc, rate_of_change, boundary_mask)

        self.tableau = tableau
        self.dudt_buffers = None
        self.cfl_number = cfl_number

    def __call__(self, u0, t, dt):
        self.allocate_buffers(u0)

        uj = u0
        for j in range(self.tableau.stages):
            F, dF = self.non_linear_equation(u0, t, dt, j)
            uj = self.non_linear_solver(F, dF, uj)
            self.bc(uj)

        u1 = u0 + dt*np.sum(self.tableau.b*self.dudt_buffers, axis=-1)
        self.bc(u1)


        return u1

    def allocate_buffers(self, u):
        """Allocate rate of change buffers if needed."""
        shape = u.shape+(self.tableau.stages,)

        if self.dudt_buffers is None or self.dudt_buffers.shape != shape:
            self.dudt_buffers = np.zeros(shape)

    def non_linear_equation(self, u0, t, dt, stage):
        """Return the non-linear equation and the Jacobian.

        Arguments:
            u0: approximation of u at time `t`.
            t: current simulation time
            dt: size of the time-step
            stage: current stage of the DIRK

        Return:
            (F, dF): `F` is the non-linear equation.
                     `dF` is the Jacobian of `F`.
        """
        F = self.NonlinearEquation(u0, t, dt, self.bc, self.rate_of_change,
                                   self.tableau, self.dudt_buffers, stage)
        dF = ApproximateJacobian(F, u0, self.epsilon)

        return F, dF

    def pick_time_step(self, u):
        return self.cfl_number * self.rate_of_change.pick_time_step(u)

    class NonlinearEquation(object):
        def __init__(self, u0, t, dt, bc, rate_of_change, tableau, dudt_buffers, stage):
            self.u0, self.t, self.dt = u0, t, dt
            self.bc = bc
            self.rate_of_change = rate_of_change
            self.tableau = tableau
            self.dudt_buffers = dudt_buffers
            self.stage = stage

            self.shape = u0.shape

            s, k = stage, self.dudt_buffers
            sm1 = max(0, s-1)
            self.sum_k = np.sum(self.tableau.a[s,:sm1]*k[...,:sm1], axis=-1)

        def __call__(self, u):
            u0, t, dt = self.u0, self.t, self.dt
            u = u.reshape(self.shape)

            a, s, k = self.tableau.a, self.stage, self.dudt_buffers

            dt_star = self.tableau.c[s]*dt
            t_star = t + dt_star

            k[...,s] = self.rate_of_change(u, t_star)
            residual =  u - (u0 + dt_star*(self.sum_k + a[s,s]*k[...,s]))

            return residual.reshape((-1))

class DIRKa23(DIRK):
    """Two stage, third order, A-stable DIRK."""

    def __init__(self, bc, rate_of_change, boundary_mask, cfl_number):
        isqrt3 = 1.0/np.sqrt(3.0)
        a11 = 0.5*(1.0 + isqrt3)
        a21 = -isqrt3
        a22 = 0.5*(1.0 + isqrt3)

        a = np.array([[a11, 0.0], [a21, a22]])
        b = np.array([0.5, 0.5])
        tableau = ButcherTableau(a, b)

        super().__init__(bc, rate_of_change, boundary_mask, cfl_number, tableau)

class DIRKa34(DIRK):
    """Three stage, fourth order, A-stable DIRK."""

    def __init__(self, bc, rate_of_change, boundary_mask, cfl_number):
        alpha = 2.0*np.cos(np.pi/18.0)/np.sqrt(3.0)
        a11 = 0.5*(1 + alpha)
        a21 = -0.5*alpha
        a22 = 0.5*(1 + alpha)
        a31 = 1 + alpha
        a32 = -(1 + 2*alpha)
        a33 = 0.5*(1 + alpha)


        a = np.array([[a11, 0.0, 0.0],
                      [a21, a22, 0.0],
                      [a31, a32, a33]])
        b = np.array([1.0/(6.0*alpha**2), 1.0 - 1.0/(3*alpha**2), 1.0/(6.0*alpha**2)])
        tableau = ButcherTableau(a, b)

        super().__init__(bc, rate_of_change, boundary_mask, cfl_number, tableau)
