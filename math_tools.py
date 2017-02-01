import numpy as np

def gaussian(x, sigma):
    return np.exp(-(x/sigma)**2)
