import numpy as np

def convergence_rate(error, resolution):
    return np.log(error[:-1]/error[1:])/np.log(resolution[:-1]/resolution[1:])

def l1_error(approximate, reference):
    """L1-error on a uniform grid."""

    err = np.mean(np.abs(approximate - reference), axis=-1)
    if approximate.ndim == 3:
        err = np.mean(err, axis = -1)

    return err
import pytest

mark_manual = pytest.mark.skipif(not pytest.config.getoption("--run-manual"),
                                 reason="pass `--run-manual` to run this")
