import numpy as np
import matplotlib.pyplot as plt

class SimplePlotting(object):
    def __init__(self, grid, base_name):
        self.grid = grid
        self.base_name = base_name
        self.n_plots = 0

    def file_name(self):
        pattern = self.base_name + "-{:04d}.png"
        return pattern.format(self.n_plots)

class SimpleGraph(SimplePlotting):
    def __call__(self, u):
        plt.plot(self.grid.cell_centers, u[:,:,0])

        # plt.savefig(self.file_name())
        self.n_plots += 1

class SimpleColormap(SimplePlotting):
    def __call__(self, u):
        plt.contourf(self.grid.X, self.grid.Y, u[:,:,0])

        # plt.savefig(self.file_name())
        self.n_plots += 1
