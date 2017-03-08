import numpy as np

def gaussian(x, sigma):
    return np.exp(-(x/sigma)**2)

def convergence_rate(error, resolution):
    return -np.log(error[...,1:]/error[...,:-1])/np.log(resolution[1:]/resolution[:-1])

def l1_error(approximate, reference):
    """L1-error on a uniform grid.

    The arguments are expected to shaped like FVM arrays:
        -- first axis if the species, rho, v, upsilon, etc.
        -- second axis is the x-axis.
        -- third axis is the optional y-axis.
    """

    err = np.mean(np.abs(approximate - reference), axis=-1)
    if approximate.ndim == 3:
        err = np.mean(err, axis = -1)

    return err

def linf_error(approximate, reference):
    """L^inf-error output of FVM.

    The arguments are expected to shaped like FVM arrays:
        -- first axis if the species, rho, v, upsilon, etc.
        -- second axis is the x-axis.
        -- third axis is the optional y-axis.
    """

    err = np.max(np.abs(approximate - reference), axis=-1)

    if approximate.ndim == 3:
        err = np.max(err, axis = -1)

    return err
