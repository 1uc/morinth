# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

from datetime import datetime

class ProgressBar:
    def __init__(self, unit_size):
        self.t_start = datetime.now()

        self.unit_size = unit_size
        self.unit_symbol = '.'

    def __call__(self, time_keeper):
        step = time_keeper.n_steps

        if step == 0:
            return

        if step % self.unit_size == 0:
            print(self.unit_symbol, end="", flush=True)

        if step % (5*self.unit_size) == 0:
            print(" ", end="", flush=True)

        if step % (10*self.unit_size) == 0:
            progress = time_keeper.progress_string()
            elapsed_time = self.elapsed_time()
            print(" {:s} ({:s}; {:d})".format(progress, elapsed_time, step))

    def welcome(self):
        date = ".".join(str(self.t_start).split(".")[:-1])
        print("--- New run ------")
        print("  Date: {:s}".format(date))
        print("------------------")

    def goodbye(self, time_keeper):
        print("\n")
        print("------------------")
        print(" Duration: {:s}".format(self.elapsed_time()))
        print("    Steps: {:d}".format(time_keeper.n_steps))
        print("--- End of run ---")

    def elapsed_time(self):
        diff = datetime.now() - self.t_start
        return ".".join(str(diff).split(".")[:-1])

class SilentProgressBar(ProgressBar):
    def __init__(self):
        anything = 10
        super().__init__(unit_size=anything)

    def __call__(self, time_keeper):
        return

    def welcome(self):
        return None

    def goodbye(self, time_keeper):
        return None
