import numpy as np

from burgers import Burgers
from rusanov import Rusanov
from flux_test import ContinuityTestSuite

import pytest

class RusanovContinuityTest(ContinuityTestSuite):
    @pytest.fixture
    def model(self):
        return Burgers()

    @pytest.fixture
    def num_flux(self, model):
        return Rusanov(model)

    @pytest.fixture
    def random_state(self):
        return np.random.random((100, 1))
