import numpy as np
from burgers import Burgers

def test_burgers():
    model = Burgers()
    u = np.random.random((100, 1))

    assert np.all(model.max_eigenvalue(u) >= 0.0)
