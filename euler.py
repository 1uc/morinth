import numpy as np

class EulerModel(object):
    """Euler equations with gravity."""

    def __init__(self, gamma, gravity, specific_gas_constant):
        self.eos = IdealGasLaw(gamma, specific_gas_constant)

        if isinstance(gravity, Gravity):
            self.gravity = gravity
        else:
            self.gravity = PointMassGravity(1.0, gravity, 1.0)

    def sound_speed(self, u, p):
        """The speed of sound in the system."""
        return self.eos.sound_speed(u[0,...], p)

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

    def temperature(self, rho, p):
        return self.eos.temperature(rho, p)

    def enthalpy(self, rho, p):
        return self.eos.enthalpy(rho, p)

    def rho(self, p, T):
        return self.eos.rho(p, T)

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

    def source(self, u, x):
        """Physical source term of the Euler equations."""
        source = np.empty_like(u)

        dphi_dx = self.gravity.dphi_dx(x)

        source[0,...] = 0.0
        source[1,...] = -u[0,...]*dphi_dx
        source[2,...] = 0.0
        source[3,...] = -u[1,...]*dphi_dx

        return source

    def pressure(self, u=None, rho=None, T=None):
        if rho is not None and T is not None:
            return self.eos.pressure(rho=rho, T=T)

        elif u is not None:
            e = self.specific_internal_energy(u)
            return self.eos.pressure(e=e, rho=u[0,...])

        else:
            raise Exception("Invalid combination of arguments.")

    def kinetic_energy(self, u):
        return 0.5*(u[1,...]**2 + u[2,...]**2)/u[0,...]

    def internal_energy(self, p):
        return self.eos.internal_energy(p)

    def specific_internal_energy(self, u):
        E_int = u[3,...] - self.kinetic_energy(u)
        return E_int / u[0,...]

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

class Gravity():
    pass

class LinearGravity(Gravity):
    def __init__(self, g):
        self.g = g

    def phi(self, x):
        return self.g*x

    def dphi_dx(self, x):
        return self.g

    def dphi_dxx(self, x):
        return 0.0

class PointMassGravity(Gravity):
    def __init__(self, mass, gravitational_constant, radius):
        self.G = gravitational_constant
        self.GM = self.G*mass
        self.radius = radius

    def phi(self, x):
        return -self.GM/(x + self.radius)

    def dphi_dx(self, x):
        return self.GM/(x + self.radius)**2

    def dphi_dxx(self, x):
        return -2.0*self.GM/(x + self.radius)**3

class EquationOfState():
    pass

class IdealGasLaw(EquationOfState):
    def __init__(self, gamma, specific_gas_constant):
        self.gamma = gamma
        self.specific_gas_constant = specific_gas_constant

    def pressure(self, e=None, rho=None, T=None):
        if T is not None and rho is not None:
            Rgas = self.specific_gas_constant
            return rho*Rgas*T

        elif e is not None and rho is not None:
            return (self.gamma-1.0)*e*rho

        else:
            raise Exception("Failed to match a mode.")

    def rho(self, p, T):
        Rgas = self.specific_gas_constant
        return p/(Rgas*T)

    def temperature(self, rho, p):
        Rgas = self.specific_gas_constant
        return p/(Rgas*rho)

    def sound_speed(self, rho, p):
        return np.sqrt(self.gamma*p/rho)

    def dp_drho2_s(self, rho, p, a=None):
        if a is None:
            a = self.sound_speed(rho, p)

        rho2 = rho*rho
        return self.gamma * (rho*a*a - p)/rho2

    def internal_energy(self, p):
        """Internal energy, not specific internal energy."""
        return p/(self.gamma - 1.0)

    def enthalpy(self, rho, p):
        return 1.0/(self.gamma - 1.0) * p/rho
