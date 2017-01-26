import numpy as np

class ENOBase(object):
    """Base class for (W)ENO reconstruction."""

    def __call__(self, u, axis):
        """Return reconstructed values U-, U+."""

        # Shuffle x-axis to the last position.
        if axis == 0 and u.ndim == 3:
            u = u.transpose((0, 2, 1))

        u_plus = self.trace_values(u)[...,:-1]
        u_minus = self.trace_values(u[...,::-1])[...,-2::-1]

        if axis == 0 and u.ndim == 3:
            u_plus = u_plus.transpose((0, 2, 1))
            u_minus = u_minus.transpose((0, 2, 1))

        return u_plus, u_minus


    def u123(self, u):
        u1 = self.stencil(u, [1.0/3.0, -7.0/6.0, 11.0/6.0], (0, -4))
        u2 = self.stencil(u, [-1.0/6.0, 5.0/6.0, 1.0/3.0], (1, -3))
        u3 = self.stencil(u, [1.0/3.0, 5.0/6.0, -1.0/6.0], (2, -2))

        return u1, u2, u3

    def smoothness_indicators(self, u):
        a = 13.0/12.0*self.stencil(u, [1.0, -2.0, 1.0], (0, -2))**2
        beta1 = a[...,:-2]  + 0.25*self.stencil(u, [1.0, -4.0,  3.0], (0, -4))**2
        beta2 = a[...,1:-1] + 0.25*self.stencil(u, [1.0,  0.0, -1.0], (1, -3))**2
        beta3 = a[...,2:]   + 0.25*self.stencil(u, [3.0, -4.0,  1.0], (2, -2))**2

        return beta1, beta2, beta3

    def stencil(self, u, values, shift):
        def get_slice(u, start, end):
            if end == 0:
                return u[...,start:]
            else:
                return u[...,start:end]

        return values[0]*get_slice(u, shift[0],   shift[1])     \
             + values[1]*get_slice(u, shift[0]+1, shift[1]+1)   \
             + values[2]*get_slice(u, shift[0]+2, shift[1]+2)


class ENO(ENOBase):
    """Third order ENO reconstruction."""

    def trace_values(self, u):
        """Compute the i+1/2 ENO trace value over the last axis."""

        u1, u2, u3 = self.u123(u)
        beta1, beta2, beta3 = self.smoothness_indicators(u)

        beta_min = np.minimum(np.minimum(beta1, beta2), beta3)

        u_rc = np.where(beta1 == beta_min, u1, 0.0)
        u_rc = np.where(beta2 == beta_min, u2, u_rc)
        u_rc = np.where(beta3 == beta_min, u3, u_rc)

        return u_rc


class WENOBase(ENOBase):
    def trace_values(self, u):
        u1, u2, u3 = self.u123(u)
        w1, w2, w3 = self.non_linear_weigths(u)

        return w1*u1 + w2*u2 + w3*u3

    def non_linear_weigths(self, u):
        r = self.non_linear_weigths_exponent()
        epsilon = self.epsilon

        gamma1, gamma2, gamma3 = self.linear_weights()
        beta1, beta2, beta3 = self.smoothness_indicators(u)

        omega1 = gamma1/(epsilon + beta1)**r
        omega2 = gamma2/(epsilon + beta2)**r
        omega3 = gamma3/(epsilon + beta3)**r

        omega_total = omega1 + omega2 + omega3
        return omega1/omega_total, omega2/omega_total, omega3/omega_total


class StableWENO(WENOBase):
    """Third order, stable WENO reconstruction.

    Reference: D. S. Balsara et al, JCP, 2009
    """

    def __init__(self):
        self.epsilon = 1e-5

    def linear_weights(self):
        return (1.0, 100.0, 1.0)

    def non_linear_weigths_exponent(self):
        return 4.0


class OptimalWENO(WENOBase):
    """Fifth order WENO reconstruction.

    Reference: C.W. Shu, SIAM Review, 2009.
    """

    def __init__(self):
        self.epsilon = 1e-6

    def linear_weights(self):
        return 1.0/10.0, 3.0/5.0, 3.0/10.0

    def non_linear_weigths_exponent(self):
        return 2.0

class PrimitiveReconstruction(object):
    def __init__(self, model, reconstruction):
        super().__init__()
        self.model = model
        self.reconstruction = reconstruction

    def __call__(self, u, axis):
        model = self.model
        w = model.primitive_variables(u)

        w_left, w_right = self.reconstruction(w, axis)

        u_left = model.conserved_variables(w_left)
        u_right = model.conserved_variables(w_right)

        return u_left, u_right


