import numpy as np
import matplotlib.pyplot as plt

class PlottingBase(object):
    def __init__(self, grid, base_name):
        self.grid = grid
        self.base_name = base_name
        self.n_plots = 0
        plt.clf()

    def __call__(self, u):
        self.plot(u)

        plt.savefig(self.file_name())
        self.n_plots += 1

    def file_name(self):
        pattern = self.base_name + "-{:04d}.png"
        return pattern.format(self.n_plots)


class SimpleGraph(PlottingBase):
    def plot(self, u):
        plt.plot(self.grid.cell_centers, u[0,:,:])


class SimpleColormap(PlottingBase):
    def plot(self, u):
        plot.contourf(self.grid.X, self.grid.Y, u[0,...])


class ColormapWithArrows(PlottingBase):
    def __init__(self, grid, base_name):
        super().__init__(grid, base_name)

        self.contourf_kwargs = {'cmap': 'viridis'}
        self.quiver_kwargs = {}


    def plot(self, u):
        plt.clf()

        scalar = self.transform_scalar(u)
        vector = self.transform_vector(u)

        X, Y = self.grid.X, self.grid.Y
        plt.contourf(X, Y, scalar, 50, **self.contourf_kwargs)
        plt.colorbar()

        quiv = plt.quiver(X[::10,::10],
                          Y[::10,::10],
                          vector[0,::10,::10],
                          vector[1,::10,::10],
                          **self.quiver_kwargs)


        # plt.quiverkey(quiv, 1, -0.18, *quiver_label(max_speed(v)))

    def transform_vector(self, u):
        return u[1:3,...] / u[0,...]


class DensityPlots(ColormapWithArrows):
    def transform_scalar(self, u):
        return u[0,...]

class LogDensityPlots(ColormapWithArrows):
    def transform_scalar(self, u):
        return np.log10(u[0,...])


class PressurePlots(ColormapWithArrows):
    def __init__(self, grid, base_name, model):
        super().__init__(grid, base_name)
        self.model = model

    def transform_scalar(self, u):
        return self.model.pressure(u)


class LogPressurePlots(PressurePlots):
    def transform_scalar(self, u):
        return np.log10(self.model.pressure(u))



class EulerPlots(PlottingBase):
    def __init__(self, grid, base_name, model):
        self.density_plots = DensityPlots(grid, base_name + "-rho")
        self.log_density_plots = LogDensityPlots(grid, base_name + "-rho")
        self.log_pressure_plots = LogPressurePlots(grid, base_name + "-p", model)

    def __call__(self, u):
        self.density_plots(u)
        self.log_density_plots(u)
        self.log_pressure_plots(u)
