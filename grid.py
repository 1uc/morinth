import numpy as np

class Grid(object):
    """Simple structured 1D grid."""

    def __init__(self, domain, n_cells, n_ghost):
        self._domain = np.array(domain)
        self._n_cells = n_cells

        self.n_dims = self._domain.ndim
        self.n_ghost = n_ghost

        self.make_boundary_mask()

        if self.n_dims == 1:
            self.make_1d_grid()
        elif self.n_dims == 2:
            self.make_2d_grid()
        else:
            raise Exception("Strange dimensions.")

    def make_boundary_mask(self):
        if isinstance(self._n_cells, int):
            shape = (1, self._n_cells, 1)
        else:
            shape = (1, self._n_cells[0], self._n_cells[1])

        n_ghost = self.n_ghost

        mask = np.full(shape, True, dtype=bool)
        mask[:,:n_ghost,:] = False
        mask[:,-n_ghost:,:] = False
        if len(shape) == 2:
            mask[:,:,:n_ghost] = False
            mask[:,:,-n_ghost:] = False

        self.boundary_mask = mask

    def make_1d_grid(self):
        x = np.linspace(self._domain[0], self._domain[1], self._n_cells+1)

        self.edges = x.reshape((-1, 1))
        self.cell_centers = 0.5*(self.edges[1:,:] + self.edges[:-1,:])

        self.dx = self.edges[1,0] - self.edges[0,0]

    def make_2d_grid(self):
        x = np.linspace(self._domain[0,0], self._domain[0,1], self._n_cells[0]+1)
        y = np.linspace(self._domain[1,0], self._domain[1,1], self._n_cells[1]+1)

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
