import numpy as np

from morinth.euler import Euler
from morinth.hllc import HLLC
from morinth.rusanov import Rusanov
from morinth.flux_test import ContinuityTestSuite

import pytest

class HLLCContinuityTest(ContinuityTestSuite):
    @pytest.fixture
    def model(self):
        return Euler(gravity = 1.0, gamma = 1.66, specific_gas_constant = 1.0)

    @pytest.fixture
    def num_flux(self, model):
        return HLLC(model)

    @pytest.fixture
    def random_state(self, model):
        state = np.random.random((4, 10, 13))
        state[0,...] = np.abs(state[0,...]) + 0.1
        state[3,...] = np.abs(state[3,...]) + 0.1
        return model.conserved_variables(state)
