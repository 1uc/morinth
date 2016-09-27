import numpy as np

class TimeLoop(object):
    def __init__(self, single_step, visualize, plotting_steps):
        self.single_step = single_step
        self.visualize = visualize
        self.plotting_steps = plotting_steps

    def __call__(self, u0, time_keeper):
        u = u0
        dt = self.pick_time_step(u, time_keeper)

        while not time_keeper.is_finished():
            u = self.single_step(u, time_keeper.t, dt)
            time_keeper.advance_by(dt)

            if self.plotting_steps.is_plotting_step(time_keeper):
                self.visualize(u)

            dt = self.pick_time_step(u, time_keeper)

        return u

    def pick_time_step(self, u, time_keeper):
        dt = self.single_step.pick_time_step(u)
        dt = self.plotting_steps.pick_time_step(time_keeper, dt)
        return time_keeper.pick_time_step(u, dt)
