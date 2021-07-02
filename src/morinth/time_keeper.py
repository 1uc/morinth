# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

class TimeKeeper(object):
    def __init__(self, needs_baby_steps):
        self.needs_baby_steps = needs_baby_steps

        self.t = 0.0
        self.n_steps = 0

    def advance_by(self, dt):
        self.t += dt
        self.n_steps += 1

    def baby_steps(self, dt):
        if self.needs_baby_steps:
            if self.n_steps < 10:
                dt *= 0.01
            elif 10 <= self.n_steps < 20:
                dt *= 0.1
            elif 20 <= self.n_steps < 30:
                dt *= 0.5

        return dt

class FixedSteps(TimeKeeper):
    def __init__(self, max_steps, needs_baby_steps=False):
        super().__init__(needs_baby_steps)
        self.max_steps = max_steps

    def is_finished(self):
        return self.n_steps >= self.max_steps

    def pick_time_step(self, u, dt):
        return self.baby_steps(dt)

    def progress_string(self):
        return "{:.2e} / ---".format(self.t)

class FixedDuration(TimeKeeper):
    def __init__(self, T, needs_baby_steps=False):
        super().__init__(needs_baby_steps)
        self.T = T

    def is_finished(self):
        return self.t >= self.T

    def pick_time_step(self, u, dt):
        return min(self.T - self.t, self.baby_steps(dt))

    def progress_string(self):
        return "{:.2e} / {:.2e}".format(self.t, self.T)

class PlotAtFixedInterval(object):
    def __init__(self, dt_vis):
        self.dt_vis = dt_vis
        self.t_vis = dt_vis
        self.t_vis_last = 0.0

    def is_plotting_step(self, time_keeper):
        t, t_vis, t_vis_last = time_keeper.t, self.t_vis, self.t_vis_last
        is_first_frame = t == 0
        is_over_due = (t >= t_vis)
        is_repeated_query = (t == t_vis_last)
        is_finished = time_keeper.is_finished()

        if is_over_due:
            self.t_vis_last = t
            self.t_vis += self.dt_vis

        return is_first_frame or is_over_due or is_repeated_query or is_finished

    def pick_time_step(self, time_keeper, dt):
        return min(dt, self.t_vis - time_keeper.t)


class PlotEveryNthStep(object):
    def __init__(self, steps_per_frame):
        self.steps_per_frame = steps_per_frame

    def is_plotting_step(self, time_keeper):
        is_due = time_keeper.n_steps % self.steps_per_frame == 0
        is_finished = time_keeper.is_finished()
        return is_due or is_finished

    def pick_time_step(self, time_keeper, dt):
        return dt

class PlotNever(object):
    def is_plotting_step(self, time_keeper):
        return False

    def pick_time_step(self, time_keeper, dt):
        return dt

class PlotLast:
    def is_plotting_step(self, time_keeper):
        return time_keeper.is_finished()

    def pick_time_step(self, time_keeper, dt):
        return dt
