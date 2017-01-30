import numpy as np

class EulerModel(object):
    """Euler equations with gravity."""

    def __init__(self, gamma, gravity, specific_gas_constant = None):
        self.gamma = gamma
        self.gravity = gravity
        self.specific_gas_constant = specific_gas_constant

    def sound_speed(self, u, p):
        """The speed of sound in the system."""
        return np.sqrt(self.gamma*p/u[0,...])

    def conserved_variables(self, primitive_variables):
        """Return the conserved variables, given the primitive variables."""
        pvars = primitive_variables
        cvars = np.empty_like(pvars)
        cvars[0,...] = pvars[0,...]
        cvars[1,...] = pvars[0,...]*pvars[1,...]
        cvars[2,...] = pvars[0,...]*pvars[2,...]
        cvars[3,...] = self.internal_energy(pvars[3,...]) + self.kinetic_energy(cvars)

        return cvars

    def primitive_variables(self, conserved_variables):
        """Return the primitive variables, given the conserved variables."""
        cvars = conserved_variables
        pvars = np.empty_like(cvars)

        pvars[0,...] = cvars[0,...]
        pvars[1,...] = cvars[1,...]/cvars[0,...]
        pvars[2,...] = cvars[2,...]/cvars[0,...]
        pvars[3,...] = self.pressure(cvars)

        return pvars

    def max_eigenvalue(self, u):
        """Largest eigenvalue of the flux derivative, df/du."""
        p = self.pressure(u)
        return self.speed(u) + self.sound_speed(u, p)



class Euler(EulerModel):
    def flux(self, u, axis, p=None):
        """Physical flux of the Euler equations."""
        if p is None:
            p = self.pressure(u)

        v = u[axis+1,...]/u[0,...]

        flux = v*u

        flux[axis+1,...] += p
        flux[3,...] += v*p

        return flux

    def source(self, u, axis):
        """Physical source term of the Euler equations."""
        source = np.empty_like(u)

        source[0,...] = 0.0
        source[1,...] = 0.0
        source[2,...] = -u[0,...]*self.gravity
        source[3,...] = -u[axis+1,...]*self.gravity

        return source

    def pressure(self, u):
        return (self.gamma-1.0)*(u[3,...] - self.kinetic_energy(u))

    def kinetic_energy(self, u):
        return 0.5*(u[1,...]**2 + u[2,...]**2)/u[0,...]

    def internal_energy(self, p):
        return p/(self.gamma-1.0)

    def speed(self, u):
        """Norm of the velocity."""
        return np.sqrt((u[1,...]**2 + u[2,...]**2))/u[0,...]

class QuasiLinearEuler(EulerModel):
    """Euler equations in quasilinear form in terms of the primitive variables.

    In quasi-linear for the equations read:
        W_t + A(W)W_x + B(W)W_y + C(W)W_z = 0
    """

    def eigenvectors(self, w, axis):
        """Return eigenvectors of Euler equation in primitive variables.

        Arguments:
            :w: Primitive variables.
            :axis: compute eigenvalues for the flux in this direction
        """
        rho = w[0,...]
        v = w[axis+1,...]
        a = self.sound_speed(w, w[3,...])
        rho_a_square = rho*a*a

        eig_vec = np.zeros((4, 4) + w.shape[1:])

        # K^1 eigenvector
        eig_vec[0,0,...] = rho
        eig_vec[axis+1,0,...] = -a
        eig_vec[3,0,...] = rho_a_square

        # K^2 eigenvector
        eig_vec[0,1,...] = 1.0
        if axis == 0:
            eig_vec[2,1,...] = w[2,...]
        else:
            eig_vec[1,1,...] = w[1,...]

        # K^3 eigenvector
        eig_vec[0,2,...] = rho
        if axis == 0:
            eig_vec[2,2,...] = 1.0
        else:
            eig_vec[1,2,...] = 1.0

        # K^4 eigenvector
        eig_vec[0,3,...] = rho
        eig_vec[axis+1,3,...] = a
        eig_vec[3,3,...] = rho_a_square

        return eig_vec

    def coefficient_matrix(self, w, axis):
        """Coefficient matrix of the Euler equation, A(W), B(W), C(W)."""
        rho = w[0,...]
        v = w[axis+1,...]
        a = self.sound_speed(w, w[3,...])
        rho_a_square = rho*a*a

        mat = np.zeros((4, 4) + w.shape[1:])

        # diagonal terms
        mat[0,0,...] = v
        mat[1,1,...] = v
        mat[2,2,...] = v
        mat[3,3,...] = v

        mat[0,axis+1,...] = rho  # Note, this term does not depend on axis.
        mat[axis+1,3,...] = 1.0/rho
        mat[3,axis+1,...] = rho_a_square

        return mat


    def eigenvalues(self, w, axis):
        """Return eigenvalues of the Euler equation in primitive variables."""

        eig_val = np.empty_like(w)

        a = self.sound_speed(w, w[3,...])
        v = w[axis+1,...]

        # Eigenvalues corresponding to the eigenvectors.
        eig_val[0,...] = v - a
        eig_val[1,...] = v
        eig_val[2,...] = v
        eig_val[3,...] = v + a

        return eig_val
