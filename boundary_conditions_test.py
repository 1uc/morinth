import numpy as np

from grid import Grid
from boundary_conditions import Periodic

import pytest

def test_periodic_bc():
    grid = Grid([0.0, 1.0], 100, 2)
    state = grid.cell_centers.reshape((1,-1,1))**2
    bc = Periodic(grid)
    old_state = np.copy(state)
    bc(state)

    assert state[:,0,:] == old_state[:,-4,:]
    assert state[:,1,:] == old_state[:,-3,:]

    assert state[:,-1,:] == old_state[:,3,:]
    assert state[:,-2,:] == old_state[:,2,:]

def test_periodic_bc():
    grid = Grid([0.0, 1.0], 100, 1)
    state = grid.cell_centers.reshape((1,-1,1))**2
    bc = Periodic(grid)
    old_state = np.copy(state)

    bc(state)

    assert state[:,0,:] == old_state[:,-2,:]
    assert state[:,-1,:] == old_state[:,1,:]

    assert state[:,1,:] == old_state[:,1,:]
    assert state[:,-2,:] == old_state[:,-2,:]

