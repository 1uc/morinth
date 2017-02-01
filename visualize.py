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
    def __init__(self, grid, base_name, back_ground=None):
        super().__init__(grid, base_name)
        if back_ground is None:
            self.back_ground = 0.0
        else:
            self.back_ground = back_ground

    def plot(self, u):
        plt.clf()

        scalar = self.transform_scalar(u)
        plt.plot(self.grid.cell_centers, scalar, self.easy_style())
        self.xlim()

    def xlim(self):
        n_ghost = self.grid.n_ghost
        plt.xlim((self.grid.edges[n_ghost], self.grid.edges[-n_ghost]))

    def transform_scalar(self, u):
        return u[0, ...] - self.back_ground

    def easy_style(self):
        return "wo"


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
        return u[0, ...] - self.back_ground


class PressureGraph(SimpleGraph):
    def __init__(self, grid, base_name, model, back_ground=None):
        super().__init__(grid, base_name, back_ground)
        self.model = model

    def transform_scalar(self, u):
        return self.model.pressure(u) - self.back_ground


class XVelocityGraph(SimpleGraph):
    def transform_scalar(self, u):
        return u[1, ...]/u[0, ...] - self.back_ground


class YVelocityGraph(SimpleGraph):
    def transform_scalar(self, u):
        return u[2, ...]/u[0, ...] - self.back_ground


class EulerGraphs(MultiplePlots):
    def __init__(self, grid, base_name, model):
        density_plot = DensityGraph(grid, base_name + "-rho")
        vx_plot = XVelocityGraph(grid, base_name + "-vx")
        pressure_plot = PressureGraph(grid, base_name + "-p", model)

        self.all_plots = [density_plot, pressure_plot, vx_plot]


class EquilibriumGraphs(MultiplePlots):
    def __init__(self, grid, base_name, model, u0):
        density_plot = DensityGraph(grid, base_name + "-rho", u0[0,...])
        vx_plot = XVelocityGraph(grid, base_name + "-vx", u0[1,...]/u0[0,...])
        pressure_plot = PressureGraph(grid, base_name + "-p", model, model.pressure(u0))

        self.all_plots = [density_plot, pressure_plot, vx_plot]
