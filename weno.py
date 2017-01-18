import numpy as np

class WENO(object):
    """Fifth order WENO reconstruction.

    Reference: C.W. Shu, SIAM Review, 2009.
    """

    def __init__(self):
        self.epsilon = 1e-6

    def weno_trace(self, u):
        """Compute the i+1/2 WENO trace value over the last axis."""

        u1 = self.stencil(u, [1.0/3.0, -7.0/6.0, 11.0/6.0], (0, -4))
        u2 = self.stencil(u, [-1.0/6.0, 5.0/6.0, 1.0/3.0], (1, -3))
        u3 = self.stencil(u, [1.0/3.0, 5.0/6.0, -1.0/6.0], (2, -2))

        a = 13.0/12.0*self.stencil(u, [1.0, -2.0, 1.0], (0, -2))**2
        beta1 = a[...,:-2]  + 0.25*self.stencil(u, [1.0, -4.0,  3.0], (0, -4))**2
        beta2 = a[...,1:-1] + 0.25*self.stencil(u, [1.0,  0.0, -1.0], (1, -3))**2
        beta3 = a[...,2:]   + 0.25*self.stencil(u, [3.0, -4.0,  1.0], (2, -2))**2

        gamma1, gamma2, gamma3 = 1.0/10.0, 3.0/5.0, 3.0/10.0

        epsilon = self.epsilon
        omega1 = gamma1/(epsilon + beta1)**2
        omega2 = gamma2/(epsilon + beta2)**2
        omega3 = gamma3/(epsilon + beta3)**2

        omega_total = omega1 + omega2 + omega3
        w1 = omega1/(omega_total)
        w2 = omega2/(omega_total)
        w3 = omega3/(omega_total)

        u_rc = w1*u1 + w2*u2 + w3*u3

        return u_rc

    def __call__(self, u, axis):
        """Return reconstructed values U-, U+."""

        # Shuffle x-axis to the last position.
        if axis == 0 and u.ndim == 3:
            u = u.transpose((0, 2, 1))

        u_plus = self.weno_trace(u)
        u_minus = self.weno_trace(u[...,::-1])[...,::-1]

        if axis == 0 and u.ndim == 3:
            u_plus = u_plus.transpose((0, 2, 1))
            u_minus = u_minus.transpose((0, 2, 1))

        return u_plus, u_minus

    def stencil(self, u, values, shift):
        def get_slice(u, start, end):
            if end == 0:
                return u[...,start:]
            else:
                return u[...,start:end]

        return values[0]*get_slice(u, shift[0],   shift[1])     \
             + values[1]*get_slice(u, shift[0]+1, shift[1]+1)   \
             + values[2]*get_slice(u, shift[0]+2, shift[1]+2)

