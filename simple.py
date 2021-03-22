#!/usr/bin/env python3
# encoding: utf-8

import numpy as np
from morinth.euler import *
from morinth.gaussian_bump import *
from morinth.equilibrium import *
from morinth.grid import *

gravity = PointMassGravity(gravitational_constant=800.0, mass=3000.0, radius=1000.0)
euler = Euler(gamma=1.4, gravity=gravity, specific_gas_constant=0.01)
grid = Grid([0.0, 500.0], 10, 3)

equilibrium = IsentropicEquilibrium(grid, euler)
ic = GaussianBumpIC(euler, equilibrium)
ic.rho_ref = 32.0
ic.p_ref = 10000.0

u = ic(grid)
x = grid.cell_centers[...,0]

H = euler.scale_height(u, x)
print(u[[0,3],...])
print(H)

p = euler.pressure(u)
c_s = euler.sound_speed(u, p)
print(c_s)
