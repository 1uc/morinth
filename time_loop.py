import numpy as np
from progress_bar import SilentProgressBar

class TimeLoop(object):
    def __init__(self, single_step, visualize, plotting_steps, progress_bar = None):
        self.single_step = single_step
        self.visualize = visualize
        self.plotting_steps = plotting_steps

        if progress_bar is None:
            self.progress_bar = SilentProgressBar()
        else:
            self.progress_bar = progress_bar

    def __call__(self, u0, time_keeper):
        self.progress_bar.welcome()
        u = u0
        dt = self.pick_time_step(u, time_keeper)

        if self.plotting_steps.is_plotting_step(time_keeper):
            self.visualize(u)

        while not time_keeper.is_finished():
            u = self.single_step(u, time_keeper.t, dt)
            time_keeper.advance_by(dt)

            if self.plotting_steps.is_plotting_step(time_keeper):
                self.visualize(u)

            dt = self.pick_time_step(u, time_keeper)

            self.progress_bar(time_keeper)

        self.progress_bar.goodbye(time_keeper)
        return u

    def pick_time_step(self, u, time_keeper):
        dt = self.single_step.pick_time_step(u)
        dt = self.plotting_steps.pick_time_step(time_keeper, dt)
        return time_keeper.pick_time_step(u, dt)
