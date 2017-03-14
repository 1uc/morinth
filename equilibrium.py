import numpy as np

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

        return rho_point, p_point, T_tilda

    def drho_dxx_per_rho(self, T, x):
        R = self.model.specific_gas_constant
        gravity = self.model.gravity
        dphi_dx = gravity.dphi_dx(x)
        dphi_dxx = gravity.dphi_dxx(x)
        drho_dxx_per_rho = 1.0/(R*T)**2 * dphi_dx**2 - 1.0/(R*T) * dphi_dxx

        return drho_dxx_per_rho

    def extrapolate(self, p_ref, T_ref, x_ref, x):
        """Extrapolate point-values in a reference point to point-values."""
        gravity, R = self.model.gravity, self.model.specific_gas_constant

        p = p_ref*np.exp(-(gravity.phi(x) - gravity.phi(x_ref))/(R*T_ref))
        rho = self.model.rho(p=p, T=T_ref)

        return rho, p

    def reconstruct(self, u_bar, x_ref, x):
        """Reconstruct the equilibrium cell-average in another cell."""
        model = self.model

        rho_point, p_point, T_tilda = self.point_values(u_bar, x_ref)
        dx2_24 = self.grid.dx**2 / 24.0
        drho_dxx_per_rho = self.drho_dxx_per_rho(T_tilda, x)

        rho_eq, p_eq = self.extrapolate(p_point, T_tilda, x_ref, x)
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
