# SPDX-License-Identifier: MIT
# Copyright (c) 2021 ETH Zurich, Luc Grosheintz-Laval

import numpy as np

class Grid(object):
    """Simple structured 1D grid."""

    def __init__(self, domain, n_cells, n_ghost):
        self._domain = np.array(domain)

        try:
            self.n_cells = tuple(n_cells)
        except:
            self.n_cells = (n_cells,)

        self.n_dims = self._domain.ndim
        self.n_ghost = n_ghost
        self.lower_boundary = np.reshape(self._domain.transpose(), (2, -1))[0,:]
        self.upper_boundary = np.reshape(self._domain.transpose(), (2, -1))[1,:]
        self.extent = self.upper_boundary - self.lower_boundary

        self.make_boundary_mask()

        if self.n_dims == 1:
            self.make_1d_grid()
        elif self.n_dims == 2:
            self.make_2d_grid()
        else:
            raise Exception("Strange dimensions.")

    def make_boundary_mask(self):
        shape = (1,) + self.n_cells

        n_ghost = self.n_ghost

        mask = np.full(shape, True, dtype=bool)
        mask[:,:n_ghost,...] = False
        mask[:,-n_ghost:,...] = False

        if self.n_dims == 2:
            mask[:,:,:n_ghost] = False
            mask[:,:,-n_ghost:] = False

        self.boundary_mask = mask

    def partition(self, x0, x1, n_cells):
        n_ghost = self.n_ghost
        x = np.empty(n_cells+1)
        x[n_ghost:n_cells+1-n_ghost] = np.linspace(x0, x1, n_cells - 2*n_ghost + 1)

        dx0 = x[n_ghost+1] - x[n_ghost]
        dx1 = x[-n_ghost-1] - x[-n_ghost-2]

        for k in range(1, n_ghost+1):
            x[n_ghost-k] = x[n_ghost] - k*dx0
            x[-n_ghost+k-1] = x[-n_ghost-1] + k*dx1

        return x

    def make_1d_grid(self):
        x = self.partition(self._domain[0], self._domain[1], self.n_cells[0])

        self.edges = x.reshape((-1, 1))
        self.cell_centers = 0.5*(self.edges[1:,:] + self.edges[:-1,:])

        self.dx = self.edges[1,0] - self.edges[0,0]

    def make_2d_grid(self):
        x = self.partition(self._domain[0,0], self._domain[0,1], self.n_cells[0])
        y = self.partition(self._domain[1,0], self._domain[1,1], self.n_cells[1])

        self.edges = np.empty((x.shape[0], y.shape[0], 2))
        X, Y = np.meshgrid(x, y)
        self.edges[:,:,0], self.edges[:,:,1] = X.transpose(), Y.transpose()

        x = 0.5*(x[1:] + x[:-1])
        y = 0.5*(y[1:] + y[:-1])

        self.cell_centers = np.empty((x.shape[0], y.shape[0], 2))
        X, Y = np.meshgrid(x, y)
        self.cell_centers[:,:,0], self.cell_centers[:,:,1] = X.transpose(), Y.transpose()
        self.X = X.transpose()
        self.Y = Y.transpose()

        self.dx = self.cell_centers[1,0,0] - self.cell_centers[0,0,0]
        self.dy = self.cell_centers[0,1,1] - self.cell_centers[0,0,1]
