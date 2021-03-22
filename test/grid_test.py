import numpy as np

from morinth.grid import Grid

import pytest

def test_grid_init():
    n_cells, n_ghost = 32, 2
    grid = Grid([2, 3], n_cells, n_ghost)

    assert grid.n_ghost == n_ghost
    assert grid.cell_centers.shape[0] == n_cells
    assert grid.edges.shape[0] == n_cells+1

def test_grid_2d():
    n_cells, n_ghost = [32, 20], 2
    grid = Grid([[2, 3], [0, 1]], n_cells, n_ghost)

    assert grid.cell_centers[0, 0, 0] <  grid.cell_centers[1, 0, 0]
    assert grid.cell_centers[0, 0, 1] == grid.cell_centers[1, 0, 1]

    assert grid.cell_centers[0, 0, 0] == grid.cell_centers[0, 1, 0]
    assert grid.cell_centers[0, 0, 1] < grid.cell_centers[0, 1, 1]
