# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

import itertools
import pickle
import numpy as np
import matplotlib.pyplot as plt

from morinth.coding_tools import with_default


class KeepInMemory:
    def __init__(self, time_keeper, snapshots):
        self.time_keeper = time_keeper
        self.snapshots = snapshots

    def __call__(self, u):
        self.snapshots.append({"t": self.time_keeper.t, "u": np.copy(u)})


class PlottingBase(object):
    def __init__(self, grid, base_name):
        self.grid = grid
        self.base_name = base_name
        self.n_plots = 0
        # plt.clf()

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
        plt.xlim((self.grid.edges[0, 0], self.grid.edges[-1, 0]))

    def transform_scalar(self, u):
        return u[0, ...] - self.back_ground

    def easy_style(self):
        return "ko"


class ColormapWithArrows(PlottingBase):
    def __init__(self, grid, base_name):
        super().__init__(grid, base_name)

        self.contourf_kwargs = {"cmap": "viridis"}
        self.quiver_kwargs = {}

    def plot(self, u):
        plt.clf()

        scalar = self.transform_scalar(u)
        vector = self.transform_vector(u)

        X, Y = self.grid.X, self.grid.Y
        plt.contourf(X, Y, scalar, 50, **self.contourf_kwargs)
        plt.colorbar()

        quiv = plt.quiver(
            X[::10, ::10],
            Y[::10, ::10],
            vector[0, ::10, ::10],
            vector[1, ::10, ::10],
            **self.quiver_kwargs
        )

        # plt.quiverkey(quiv, 1, -0.18, *quiver_label(max_speed(v)))

    def transform_vector(self, u):
        return u[1:3, ...] / u[0, ...]


class DensityColormap(ColormapWithArrows):
    def transform_scalar(self, u):
        return u[0, ...]


class LogDensityColormap(ColormapWithArrows):
    def transform_scalar(self, u):
        return np.log10(u[0, ...])


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
        return u[1, ...] / u[0, ...] - self.back_ground


class YVelocityGraph(SimpleGraph):
    def transform_scalar(self, u):
        return u[2, ...] / u[0, ...] - self.back_ground


class EulerGraphs(MultiplePlots):
    def __init__(self, grid, base_name, model):
        density_plot = DensityGraph(grid, base_name + "-rho")
        vx_plot = XVelocityGraph(grid, base_name + "-vx")
        pressure_plot = PressureGraph(grid, base_name + "-p", model)

        self.all_plots = [density_plot, pressure_plot, vx_plot]


class EquilibriumGraphs(MultiplePlots):
    def __init__(self, grid, base_name, model, u0):
        density_plot = DensityGraph(grid, base_name + "-rho", u0[0, ...])
        vx_plot = XVelocityGraph(grid, base_name + "-vx", u0[1, ...] / u0[0, ...])
        pressure_plot = PressureGraph(grid, base_name + "-p", model, model.pressure(u0))

        self.all_plots = [density_plot, pressure_plot, vx_plot]


class Markers:
    def __init__(self, markers=None):
        self.markers = with_default(markers, ["^", "v", "<", ">", "+", "x"])
        self.sequence = itertools.cycle(self.markers)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.sequence)


class CombineIO:
    def __init__(self, *args):
        self.all_io_objects = args

    def __call__(self, u):
        for obj in self.all_io_objects:
            obj(u)


class ConvergencePlot:
    def __init__(self, trend_orders=None):
        self.trend_orders = with_default(trend_orders, [])

    def __call__(self, all_errors, resolution, all_labels):
        plt.clf()
        marker = Markers()

        for error, label in zip(all_errors, all_labels):
            plt.plot(resolution, error, marker=next(marker), label=label)

        self.trend_line(all_errors, resolution)

        plt.yscale("log")
        plt.xscale("log")

        self.xlabel()
        self.ylabel()

        plt.legend(loc="best")

    def trend_line(self, all_errors, resolution_):
        resolution = resolution_.astype(float)
        n, N = resolution[0], resolution[-1]
        r = self.trend_orders[-1]

        min_error = min([np.min(err) for err in all_errors])
        x0 = min_error * N ** r

        for k, rate in enumerate(self.trend_orders):
            offset = 2.0 + 1.0 * (len(self.trend_orders) - k)
            errors = offset * x0 * n ** (rate - r) * resolution ** -rate
            plt.plot(resolution, errors, "k-", label="$O(N^{{-{:d}}})$".format(rate))

    def save(self, filename_base):
        plt.savefig(filename_base + ".eps")
        plt.savefig(filename_base + ".png")

    def show(self):
        plt.show()

    def xlabel(self):
        plt.xlabel("Resolution")

    def ylabel(self):
        plt.ylabel("Error")


class DumpToDisk:
    def __init__(self, grid, base_name, background=None):
        self.filename_pattern = base_name + "_data-{:04d}.npy"
        self.current_snapshot = 0

        self.save_grid(grid, base_name)

        if background is not None:
            self.save_background(background, base_name)

    def __call__(self, u):
        filename = self.filename_pattern.format(self.current_snapshot)
        np.save(filename, u)

        self.current_snapshot += 1

    def save_grid(self, grid, base_name):
        filename = base_name + "_grid.pkl"
        with open(filename, "wb") as f:
            pickle.dump(grid, f)

    def save_background(self, background, base_name):
        filename = base_name + "_background.npy"
        np.save(filename, background)


class DumpToDiskWithTime(DumpToDisk):
    def __init__(self, grid, base_name, time_keeper, background=None):
        super().__init__(grid, base_name, background)

        self.time_keeper = time_keeper
        self.filename_pattern_time = base_name + "_time-{:04d}.npy"

    def __call__(self, u):
        filename = self.filename_pattern_time.format(self.current_snapshot)
        np.save(filename, self.time_keeper.t)

        super().__call__(u)
