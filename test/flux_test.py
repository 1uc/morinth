import numpy as np
import pytest

class ContinuityTestSuite:
    def test_continuity(self, model, num_flux, random_state):
        u = random_state
        assert np.max(np.abs(num_flux(u, u, 0) - model.flux(u, 0))) <= 1e-12
