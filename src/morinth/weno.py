import numpy as np

class ENOBase(object):
    """Base class for (W)ENO reconstruction."""

    def __init__(self, transform=None):
        self._transform = transform
        self._small_polynomials = {}

    def __call__(self, u, axis):
        """Compute the reconstructed values U+, U-.

        The reconstructed values
            >>> u_plus, u_minus = eno(u, axis=0)

        are high-order interpolations at `edges[3:-3,...,0]` from the
        left/right respectively.

        Parameters
        ----------
               u : array_like
                   cell-averages of the conserved variables over the entire
                   domain

            axis : int
                   axis along which to perform the reconstruction

        Returns
        -------
        array_like, array_like
            (W)ENO interpolation at the cell boundary from the cell
            left/right of the interface.
        """

        # Shuffle x-axis to the last position.
        if axis == 1 and u.ndim == 3:
            raise Exception("No, 2D WENO should not work, yet.")

        u_plus = self.trace_values(u, x_rel=0.5)[:,:-1,...]
        u_minus = self.trace_values(u, x_rel=-0.5)[:,1:,...]

        return u_plus, u_minus

    def u123(self, u_pack, x):
        ua, ub, uc, ud, ue = u_pack
        p = self.small_polynomials(x)
        u1 = self.stencil(ua, ub, uc, p[0,:])
        u2 = self.stencil(ub, uc, ud, p[1,:])
        u3 = self.stencil(uc, ud, ue, p[2,:])

        return u1, u2, u3

    def smoothness_indicators(self, u_pack):
        ua, ub, uc, ud, ue = u_pack
        beta1 = 13.0/12.0*self.stencil(ua, ub, uc, [1.0, -2.0,  1.0])**2 \
                   + 0.25*self.stencil(ua, ub, uc, [1.0, -4.0,  3.0])**2

        beta2 = 13.0/12.0*self.stencil(ub, uc, ud, [1.0, -2.0,  1.0])**2 \
                   + 0.25*self.stencil(ub, uc, ud, [1.0,  0.0, -1.0])**2

        beta3 = 13.0/12.0*self.stencil(uc, ud, ue, [1.0, -2.0,  1.0])**2 \
                   + 0.25*self.stencil(uc, ud, ue, [3.0, -4.0,  1.0])**2

        return beta1, beta2, beta3

    def stencil(self, u1, u2, u3, alpha):
        return alpha[0]*u1 + alpha[1]*u2 + alpha[2]*u3

    def pre_transform(self, u):
        if self._transform is None:
            return u
        else:
            return self._transform.pre(u)

    def post_transform(self, u_transformed, u_rc, x_rel):
        if self._transform is None:
            return u_rc
        else:
            return self._transform.post(u_transformed, u_rc, x_rel)

    def five_stencil(self, u):
        if self._transform is None:
            return u[:,:-4], u[:,1:-3], u[:,2:-2], u[:,3:-1], u[:,4:]
        else:
            return self._transform.stencil(u)

    def small_polynomials(self, x_rel):
        """Return the coefficients of the low-order polynomials."""
        if x_rel in self._small_polynomials:
            return self._small_polynomials[x_rel]

        p, _ = self.compute_pq(x_rel)
        self._small_polynomials.update({x_rel: p})

        return self._small_polynomials[x_rel]

    def compute_pq(self, x_rel):
        L = LagrangePolynomials()
        xi = np.arange(-3, 3) + 0.5

        p = np.empty((3, 3))
        for i in range(p.shape[0]):
            p[i,:] = L.fbar_coefficients(x_rel, xi[i:i+4])

        q = L.fbar_coefficients(x_rel, xi)

        return p, q

class ENO(ENOBase):
    """Third order ENO reconstruction."""

    def trace_values(self, u, x_rel):
        """Compute the i+1/2 ENO trace value over the last axis."""
        u_transformed = self.pre_transform(u)
        u_stencil = self.five_stencil(u_transformed)
        u1, u2, u3 = self.u123(u_stencil, x_rel)
        beta1, beta2, beta3 = self.smoothness_indicators(u_stencil)

        beta_min = np.minimum(np.minimum(beta1, beta2), beta3)

        u_rc = np.where(beta1 == beta_min, u1, 0.0)
        u_rc = np.where(beta2 == beta_min, u2, u_rc)
        u_rc = np.where(beta3 == beta_min, u3, u_rc)

        return self.post_transform(u_transformed, u_rc, x_rel)

class WENOBase(ENOBase):
    def trace_values(self, u, x_rel):
        u_transformed = self.pre_transform(u)
        u_stencil = self.five_stencil(u_transformed)
        u1, u2, u3 = self.u123(u_stencil, x_rel)
        w1, w2, w3 = self.non_linear_weights(u_stencil, x_rel)

        u_rc = w1*u1 + w2*u2 + w3*u3
        return self.post_transform(u_transformed, u_rc, x_rel)

    def non_linear_weights(self, u, x_rel):
        r = self.non_linear_weights_exponent()
        epsilon = self.epsilon

        gamma1, gamma2, gamma3 = self.linear_weights(x_rel)
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

    def __init__(self, transform = None):
        super().__init__(transform)
        self.epsilon = 1e-5

    def linear_weights(self):
        return (1.0, 100.0, 1.0)

    def non_linear_weights_exponent(self):
        return 4.0


class OptimalWENO(WENOBase):
    """Fifth order WENO reconstruction.

    Reference: C.W. Shu, SIAM Review, 2009.
    """

    def __init__(self, transform = None):
        super().__init__(transform)
        self.epsilon = 1e-6
        self._linear_weights = {0.0: (1, 100, 1)}

    def linear_weights(self, x_rel):
        """Linear weights of WENO reconstruction.

        The linear weights are computed when fist needed and stored under the
        identifier `x_rel` for later reuse. Therefore, `x_rel` should be
        thought of as an identifier of the position.

        Parameters:
            x_rel : float
                Position of the reconstruction point relative
                to the cell-center in units of the mesh-width 'dx'.
        """
        if x_rel in self._linear_weights:
            return self._linear_weights[x_rel]

        else:
            return self.compute_linear_weights(x_rel)

    def compute_linear_weights(self, x_rel):
        if x_rel not in self._linear_weights:
            p, q = self.compute_pq(x_rel)

            A = np.array([[p[0,0],      0,      0],
                          [p[0,1], p[1,0],      0],
                          [p[0,2], p[1,1], p[2,0]],
                          [     0, p[1,2], p[2,1]],
                          [     0,      0, p[2,2]]])

            # The system is over-determined, solve a subsystem
            C = np.linalg.solve(A[[0, 3, 4], :], q[[0, 3, 4]])

            # .. and check its a solution of the whole system:
            assert np.all(np.abs(np.dot(A, C) - q) < 1e-12)

            self._linear_weights.update({x_rel : tuple(C)})

        return self._linear_weights[x_rel]

    def non_linear_weights_exponent(self):
        return 2.0

class LagrangePolynomials:
    def __call__(self, x, xi, k):
        deg = self.degree(xi)

        L = 1.0
        for j in range(deg):
            if j != k:
                L *= x[j]

        return L

    def degree(self, xi):
        return xi.shape[0]

    def alpha(self, xi, k):
        deg = self.degree(xi)
        return np.prod([xi[k] - xi[l] for l in range(deg) if l != k])

    def prime(self, x, xi, k):
        deg = self.degree(xi)

        def product_term(j):
            return np.prod([x - xi[l] for l in range(deg) if l != j and l !=k])

        L_prime = np.sum([product_term(j) for j in range(deg) if j != k])

        return L_prime/self.alpha(xi, k)

    def fbar_coefficients(self, x, xi):
        deg = self.degree(xi)

        L_prime = np.empty(deg)
        for k in range(deg):
            L_prime[k] = self.prime(x, xi, k)

        p = np.empty(deg-1)
        for k in range(deg-1):
            p[k] = np.sum(L_prime[k+1:])

        return p

class EquilibriumStencil(object):
    """Convert the output of an equilibrium transform into 5-stencils."""

    def __init__(self, grid, equilibrium, model):
        self.grid = grid
        self.equilibrium = equilibrium
        self.model = model

    def pre(self, u):
        return u

    def stencil(self, u):
        n_ghost = self.grid.n_ghost
        x = self.grid.cell_centers[...,0]

        u_ref = u[:,2:-2,...]
        x_ref = x[2:-2,...]

        ua = self.equilibrium.delta(u_ref, x_ref, u[:, :-4,...], x[ :-4,...])
        ub = self.equilibrium.delta(u_ref, x_ref, u[:,1:-3,...], x[1:-3,...])
        uc = self.equilibrium.delta(u_ref, x_ref, u[:,2:-2,...], x[2:-2,...])
        ud = self.equilibrium.delta(u_ref, x_ref, u[:,3:-1,...], x[3:-1,...])
        ue = self.equilibrium.delta(u_ref, x_ref, u[:,4:  ,...], x[4:  ,...])

        return ua, ub, uc, ud, ue

    def post(self, u, duij, x_rel):
        n_ghost = self.grid.n_ghost
        cell_centers = self.grid.cell_centers
        dx = self.grid.dx

        x_ref = cell_centers[2:-2,...,0]
        rho_ref, p_ref = self.equilibrium.point_values(u[:,2:-2,...], x_ref)

        xij = x_ref + x_rel*dx
        uij = np.zeros_like(duij)
        uij[0,...], p_ij = self.equilibrium.extrapolate(rho_ref, p_ref, x_ref, xij)
        uij[3,...] = self.model.internal_energy(p_ij)

        return uij + duij
