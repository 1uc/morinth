import numpy as np

from euler import Euler
from hllc import HLLC
from rusanov import Rusanov
from finite_volume_fluxes import FiniteVolumeFluxesO1
from runge_kutta import ForwardEuler
from progress_bar import ProgressBar

class EulerExperiment(object):
    """Fixture for numerical experiments with Euler equations."""

    def __init__(self):
        self.model = Euler(gravity = self.gravity(),
                           gamma = self.gamma(),
                           specific_gas_constant = self.specific_gas_constant())

        self.set_up_grid()
        self.set_up_visualization()
        self.set_up_boundary_condition()
        self.set_up_progress_bar()

    def gravity(self):
        return 0.0

    def gamma(self):
        return 1.2

    def specific_gas_constant(self):
        return 1.0

    def set_up_progress_bar(self):
        self.progress_bar = ProgressBar(10)


def prefer_hllc(model):
    """Pick a suitable numerical flux."""

    if isinstance(model, Euler):
        return HLLC(model)
    else:
        return Rusanov(model)

def scheme_o1(model, grid, bc):
    flux = prefer_hllc(model)
    fvm = FiniteVolumeFluxesO1(grid, flux)
    single_step = ForwardEuler(bc, fvm)

    return flux, fvm, single_step
