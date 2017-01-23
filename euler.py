import numpy as np

class Euler(object):
    """Euler equations with gravity."""

    def __init__(self, gamma, gravity, specific_gas_constant):
        self.gamma = gamma
        self.gravity = gravity
        self.specific_gas_constant = specific_gas_constant

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

    def max_eigenvalue(self, u):
        """Largest eigenvalue of the flux derivative, df/du."""
        p = self.pressure(u)
        return self.speed(u) + self.sound_speed(u, p)

    def pressure(self, u):
        return (self.gamma-1.0)*(u[3,...] - self.kinetic_energy(u))

    def kinetic_energy(self, u):
        return 0.5*(u[1,...]**2 + u[2,...]**2)/u[0,...]

    def internal_energy(self, p):
        return p/(self.gamma-1.0)

    def speed(self, u):
        """Norm of the velocity."""
        return np.sqrt((u[1,...]**2 + u[2,...]**2))/u[0,...]

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

