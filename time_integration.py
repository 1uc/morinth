import numpy as np

class TimeIntegration(object):
    """Interface for time-integration."""

    def __init__(self, grid, bc, rate_of_change):
        self.bc = bc
        self.rate_of_change = rate_of_change
        self.grid = grid

class ForwardEuler(TimeIntegration):
    """Simple forward Euler time-integration."""

    def __init__(self, grid, bc, rate_of_change):
        super().__init__(grid, bc, rate_of_change)
        self.cfl_number = 0.45

    def __call__(self, u0, t, dt):
        u1 = u0 + dt*self.rate_of_change(u0, t)
        self.bc(u1)

        return u1

    def pick_time_step(self, u):
        return self.cfl_number * self.rate_of_change.pick_time_step(u)

