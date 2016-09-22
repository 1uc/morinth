import numpy as np

class TimeLoop(object):
    def __init__(self, single_step, visualize):
        self.single_step = single_step
        self.visualize = visualize

    def __call__(self, u0, T):
        u = u0
        t = 0.0
        dt = self.single_step.pick_time_step(u)
        dt = np.min(dt, T - t);

        n_steps = 0

        while t < T:
            u = self.single_step(u, t, dt)
            t = t + dt

            if n_steps % 10 == 0:
                self.visualize(u)

            dt = self.single_step.pick_time_step(u)
            dt = np.min(dt, T - t);

            n_steps += 1

        return u
