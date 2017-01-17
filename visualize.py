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
        scalar = self.transform_scalar(u)
        plt.plot(self.grid.cell_centers, scalar)

    def transform_scalar(self, u):
        return u[0, :, :]


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


class DensityColormap(ColormapWithArrows):
    def transform_scalar(self, u):
        return u[0,...]


class LogDensityColormap(ColormapWithArrows):
    def transform_scalar(self, u):
        return np.log10(u[0,...])


class PressureColormap(ColormapWithArrows):
    def __init__(self, grid, base_name, model):
        super().__init__(grid, base_name)
        self.model = model

    def transform_scalar(self, u):
        return self.model.pressure(u)


class LogPressureColormap(PressureColormap):
    def transform_scalar(self, u):
        return np.log10(self.model.pressure(u))

class MultiplePlots(PlottingBase):
    def __call__(self, u):
        for plot in self.all_plots:
            plot(u)

class EulerColormaps(MultiplePlots):
    def __init__(self, grid, base_name, model):
        density_plot = DensityColormap(grid, base_name + "-rho")
        log_density_plot = LogDensityColormap(grid, base_name + "-logrho")
        log_pressure_plot = LogPressureColormap(grid, base_name + "-logp", model)

        self.all_plots = [density_plot, log_density_plot, log_pressure_plot]


class DensityGraph(SimpleGraph):
    def transform_scalar(self, u):
        return u[0, ...]

class PressureGraph(SimpleGraph):
    def __init__(self, grid, base_name, model):
        super().__init__(grid, base_name)
        self.model = model

    def transform_scalar(self, u):
        return self.model.pressure(u)

class EulerGraphs(MultiplePlots):
    def __init__(self, grid, base_name, model):
        density_plot = DensityGraph(grid, base_name + "-rho")
        pressure_plot = PressureGraph(grid, base_name + "-rho", model)

        self.all_plots = [density_plot, pressure_plot]
