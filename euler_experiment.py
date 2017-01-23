import numpy as np

from euler import Euler
from finite_volume_fluxes import FiniteVolumeFluxes
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
        return 1.4

    def specific_gas_constant(self):
        return 1.0

    def set_up_progress_bar(self):
        self.progress_bar = ProgressBar(10)
