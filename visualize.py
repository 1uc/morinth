import numpy as np
import matplotlib.pyplot as plt

class SimpleGraph(object):
    def __init__(self, grid, base_name):
        self.grid = grid
        self.base_name = base_name
        self.n_plots = 0

    def __call__(self, u):
        if self.grid.n_dims == 1:
            self.plot_1d(u)
        elif self.grid.n_dims == 2:
            self.plot_2d(u)
        else:
            raise Exception("Strange number of dimensions.")

        plt.savefig(self.file_name())

        self.n_plots += 1

    def plot_1d(self, u):
        plt.plot(self.grid.cell_centers, u[:,:,0])

    def plot_2d(self, u):
        plt.contourf(self.grid.X, self.grid.Y, u[:,:,0])

    def file_name(self):
        pattern = self.base_name + "-{:04d}.png"
        return pattern.format(self.n_plots)
