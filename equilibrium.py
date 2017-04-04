import numpy as np
import scipy
from algebraic_solvers import fixpoint_iteration

class IsothermalEquilibrium:
    """Equilibrium reconstruction of point-values."""

    def __init__(self, grid, model):
        self.grid = grid
        self.model = model

    def cell_averages(self, u_point, x):
        """Compute fourth-order equilibrium cell-average.

        Input:
        -----
        u_point : ndarray
                  Array of point values of the conserved variables
                  corresponding to position `x`.

              x : ndarray
                  Coordinates of the point values `u_point`.
        """
        model = self.model

        rho = u_point[0,...]
        p = model.pressure(u=u_point)
        T = model.temperature(rho=rho, p=p)
        drho_dxx_per_rho = self.drho_dxx_per_rho(T, x)
        dx2_24 = self.grid.dx**2 / 24.0

        rho_bar = rho*(1.0 + dx2_24*drho_dxx_per_rho)
        p_bar = model.pressure(rho=rho_bar, T=T)

        u_bar = np.empty_like(u_point)
        u_bar[0,...] = rho_bar
        u_bar[1,...] = u_point[1,...]
        u_bar[2,...] = u_point[2,...]
        u_bar[3,...] = model.internal_energy(p=p_bar)

        return u_bar

    def point_values(self, u_bar, x_mid):
        """Convert cell-averages into equilibrium point-values.

        Input:
        ------
        u_bar : ndarray
                Cell-averages of the conserved variables with
                cell-centers `x_mid`.

        x_mid : ndarray
                Coordinates of the cell-center of the cell-averages
                `u_bar`.
        """
        model = self.model

        # Convention: bar   -- cell-averages
        #             point -- high-order point-values
        #             tilda -- second-order point-values

        p_tilda = model.pressure(u_bar)
        T_tilda = model.temperature(rho=u_bar[0,...], p=p_tilda)

        dx2_24 = self.grid.dx*self.grid.dx/24.0

        drho_dxx_per_rho = self.drho_dxx_per_rho(T_tilda, x_mid)
        rho_point = u_bar[0,...]/(1 + dx2_24*drho_dxx_per_rho)
        p_point = model.pressure(rho=rho_point, T=T_tilda)

        return rho_point, p_point

    def drho_dxx_per_rho(self, T, x):
        R = self.model.eos.specific_gas_constant
        gravity = self.model.gravity
        dphi_dx = gravity.dphi_dx(x)
        dphi_dxx = gravity.dphi_dxx(x)
        drho_dxx_per_rho = 1.0/(R*T)**2 * dphi_dx**2 - 1.0/(R*T) * dphi_dxx

        return drho_dxx_per_rho

    def extrapolate(self, rho_ref, p_ref, x_ref, x):
        """Extrapolate point-values in a reference point to point-values."""
        gravity, R = self.model.gravity, self.model.eos.specific_gas_constant
        T_ref = self.model.eos.temperature(rho=rho_ref, p=p_ref)

        p = p_ref*np.exp(-(gravity.phi(x) - gravity.phi(x_ref))/(R*T_ref))
        rho = self.model.rho(p=p, T=T_ref)

        return rho, p

    def reconstruct(self, u_bar, x_ref, x):
        """Reconstruct the equilibrium cell-average in another cell."""
        model = self.model

        rho_point, p_point = self.point_values(u_bar, x_ref)
        T_tilda = model.eos.temperature(rho=rho_point, p=p_point)
        dx2_24 = self.grid.dx**2 / 24.0
        drho_dxx_per_rho = self.drho_dxx_per_rho(T_tilda, x)

        rho_eq, p_eq = self.extrapolate(rho_point, p_point, x_ref, x)
        E_int_eq = model.internal_energy(p=p_eq)

        rho_eq_bar = rho_eq*(1.0 + dx2_24*drho_dxx_per_rho)
        p_eq_bar = model.pressure(rho=rho_eq_bar, T=T_tilda)
        E_int_eq_bar = model.internal_energy(p=p_eq_bar)

        u_wb = np.empty((4,) + x.shape)
        u_wb[0,...] = rho_eq_bar
        u_wb[1,...] = 0.0
        u_wb[2,...] = 0.0
        u_wb[3,...] = E_int_eq_bar

        return u_wb

    def delta(self, u_ref, x_ref, u, x):
        """Compute the difference from equilibrium."""
        du = u - self.reconstruct(u_ref, x_ref, x)
        return du


class IsentropicEquilibrium:
    def __init__(self, grid, model):
        self.grid = grid
        self.model = model

    def cell_averages(self, u_point, x):
        u_bar = np.empty_like(u_point)
        dx = self.grid.dx
        dx2_24 = dx*dx/24.0

        drho_dxx, dE_dxx = self.du_dxx(u_point, x)

        u_bar[0,...] = u_point[0,...] + dx2_24*drho_dxx
        u_bar[1,...] = u_point[1,...]
        u_bar[2,...] = u_point[2,...]
        u_bar[3,...] = u_point[3,...] + dx2_24*dE_dxx

        return u_bar

    def point_values(self, u_bar, x):
        if u_bar.ndim == 1:
            u_bar = u_bar.reshape((-1,1))
            x = np.array([x])

        dx = self.grid.dx
        dx2_24 = dx*dx/24.0

        u_star = np.empty_like(u_bar)
        u_star[1,...] = u_bar[1,...]
        u_star[2,...] = u_bar[2,...]

        def F(rho_E_star):
            u_star[0,...] = rho_E_star[0,...]
            u_star[3,...] = rho_E_star[1,...]

            drho_dxx, dE_dxx = self.du_dxx(u_star, x)

            rho_E = np.empty((2,) + u_star.shape[1:])
            rho_E[0] = u_bar[0,...] - dx2_24*drho_dxx
            rho_E[1] = u_bar[3,...] - dx2_24*dE_dxx

            return rho_E

        rho_E, info = fixpoint_iteration(F, u_bar[[0,3],...], full_output=True)

        u_point = np.zeros_like(u_bar)
        u_point[0,...] = rho_E[0,...]
        u_point[3,...] = rho_E[1,...]

        # print(info)

        p_point = self.model.pressure(u_point)
        return u_point[0,...], p_point

    def du_dxx(self, u, x):
        rho = u[0,...]
        rho2 = rho*rho
        rho4 = rho2*rho2

        e = self.model.specific_internal_energy(u)
        p = self.model.eos.pressure(rho=rho, e=e)
        a = self.model.sound_speed(u, p)
        a2 = a*a
        a4 = a2*a2

        dphi_dx = self.model.gravity.dphi_dx(x)
        dphi_dxx = self.model.gravity.dphi_dxx(x)
        dphi_dx_2 = dphi_dx*dphi_dx

        dp_drho2_s = self.model.eos.dp_drho2_s(rho, p, a)

        drho_dx = - rho/a2 * dphi_dx
        drho_dx_2 = drho_dx * drho_dx
        drho_dxx = (1.0/rho - 1.0/a2*dp_drho2_s)*drho_dx_2 - rho/a2 * dphi_dxx

        dp_dx = -rho*dphi_dx
        dp_dxx = -(drho_dx*dphi_dx + rho*dphi_dxx)

        # d(p/rho)/dx * rho**2
        dp_rho_dx = rho*dp_dx - p*drho_dx
        dp_rho_dxx = rho*dp_dxx - p*drho_dxx

        de_dx = -dphi_dx - dp_rho_dx/rho2
        de_dxx = -dphi_dxx - (rho2*dp_rho_dxx - 2.0*rho*drho_dx*dp_rho_dx)/rho4

        dE_dxx = e*drho_dxx + 2*drho_dx * de_dx + rho*de_dxx

        return drho_dxx, dE_dxx

    def extrapolate(self, rho_ref, p_ref, x_ref, x):
        """Extrapolate point-value at `x_ref` to point-value at `x`."""
        h_ref = self.model.enthalpy(rho_ref, p_ref)

        # FIXME this assumes ideal gas, which is not appropriate.
        #    The inverting of the enthalpy should be done separately,
        #    elsewhere.
        gamma = self.model.eos.gamma
        K = p_ref/rho_ref**gamma

        rho = (1/K * (gamma-1)/gamma * self.enthalpy(h_ref, x_ref, x))**(1.0/(gamma-1))
        p = K * rho**gamma

        return rho, p

    def reconstruct(self, u_bar, x_ref, x):
        """Reconstruct cell-average at `x_ref` to cell-average at `x`.

        Given the cell-averages at `x_ref` compute the corresponding
        cell-average of the isentropic hydrostatic equilibrium at `x`.
        """
        rho_point, p_point = self.point_values(u_bar, x_ref)
        rho, p = self.extrapolate(rho_point, p_point, x_ref, x)

        w_eq = np.zeros((u_bar.shape[0],) + x.shape)
        w_eq[0,...] = rho
        w_eq[3,...] = p
        u_eq = self.model.conserved_variables(w_eq)

        return self.cell_averages(u_eq, x)

    def delta(self, u_ref, x_ref, u, x):
        return u - self.reconstruct(u_ref, x_ref, x)

    def enthalpy(self, h_ref, x_ref, x):
        gravity = self.model.gravity
        return h_ref + gravity.phi(x_ref) - gravity.phi(x)
