import numpy as np

from burgers import Burgers
from rusanov import Rusanov

import pytest

@pytest.fixture
def model():
    return Burgers()

@pytest.fixture
def rusanov(model):
    return Rusanov(model)

@pytest.fixture
def random_state():
    return np.random.random((100, 1))


def test_continuity(model, rusanov, random_state):
    u = random_state
    assert np.max(np.abs(rusanov(u, u, 0) - model.flux(u, 0))) <= 1e-12
