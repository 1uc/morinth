import numpy as np

class TimeLoop(object):
    def __init__(self, single_step, visualize):
        self.single_step = single_step
        self.visualize = visualize

    def __call__(self, u0, T):
        u = u0
        t, n_steps = 0.0, 0
        dt = self.pick_time_step(u, t, T)

        while t < T:
            u = self.single_step(u, t, dt)
            t = t + dt

            if n_steps % 10 == 0:
                self.visualize(u)

            dt = self.pick_time_step(u, t, T)

            n_steps += 1

        return u

    def pick_time_step(self, u, t, T):
        dt = self.single_step.pick_time_step(u)
        return min(dt, T - t);
