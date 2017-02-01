import numpy as np

class CenteredSourceTerm:
    def __init__(self, model):
        self.model = model

    def __call__(self, u, t):
        return self.model.source(u, t)


