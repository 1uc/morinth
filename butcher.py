import numpy as np

class ButcherTableau(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b
        self.c = np.sum(a, axis=1)
        self.stages = a.shape[0]

    def __repr__(self):
        return "".join([type(self).__name__, "(",
                        "a = ", repr(self.a), ", ",
                        "b = ", repr(self.b), ", ",
                        ")"])

    def is_explicit(self):
        """Does the tableau describe and explicit RK scheme?"""
        for i in range(self.stages):
            if not np.all(self.a[i,i:] == np.zeros((1, self.stages - i))):
                return False

        return True

    def is_implicit(self):
        return not self.is_explicit()

