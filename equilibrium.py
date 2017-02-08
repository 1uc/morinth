import numpy as np

class IsothermalEquilibrium:
    """Equilibrium reconstruction of point-values."""

    def __init__(self, grid, model):
        self.grid = grid
        self.model = model

    def cell_averages(self, u_point):
        model = self.model

        rho = u_point[0,...]
        p = model.pressure(u=u_point)
        T = model.temperature(rho=rho, p=p)
        H = model.scale_height(T)
        iH2 = 1.0/(H*H)
        dx2_24 = self.grid.dx * self.grid.dx / 24.0

        rho_bar = rho*(1.0 + dx2_24*iH2)
        p_bar = model.pressure(rho=rho_bar, T=T)

        u_bar = np.empty_like(u_point)
        u_bar[0,...] = rho_bar
        u_bar[1,...] = u_point[1,...]
        u_bar[2,...] = u_point[2,...]
        u_bar[3,...] = model.internal_energy(p=p_bar)

        return u_bar

    def point_values(self, u_bar_ref):
        model = self.model

        p_tilda_ref = model.pressure(u_bar_ref)
        T_ref = model.temperature(rho=u_bar_ref[0,...], p=p_tilda_ref)
        H = model.scale_height(T_ref)
        iH2 = 1.0/(H*H)
        dx2_24 = self.grid.dx*self.grid.dx/24.0

        rho_point = u_bar_ref[0,...]/(1 + dx2_24*iH2)
        p_point = model.pressure(rho=rho_point, T=T_ref)

        return rho_point, p_point, T_ref, iH2, dx2_24

    def extrapolate(self, p_ref, T_ref, x_ref, x):
        scale_height = self.model.scale_height(T_ref)

        p = p_ref*np.exp(-(x - x_ref)/scale_height)
        rho = self.model.rho(p=p, T=T_ref)

        return rho, p

    def reconstruct(self, u_bar_ref, x_ref, x, is_reversed=False):
        model = self.model

        rho_point, p_point, T_ref, iH2, dx2_24 = self.point_values(u_bar_ref)

        rho_eq, p_eq = self.extrapolate(p_point, T_ref, x_ref, x)
        E_int_eq = model.internal_energy(p=p_eq)

        rho_eq_bar = rho_eq + dx2_24*rho_eq*iH2
        p_eq_bar = model.pressure(rho=rho_eq_bar, T=T_ref)
        E_int_eq_bar = model.internal_energy(p=p_eq_bar)

        u_wb = np.empty((4,) + x.shape)
        u_wb[0,...] = rho_eq_bar
        u_wb[1,...] = 0.0
        u_wb[2,...] = 0.0
        u_wb[3,...] = E_int_eq_bar

        return u_wb

    def delta(self, u_ref, x_ref, u, x, is_reversed):
        """Compute the difference from equilibrium."""
        du = u - self.reconstruct(u_ref, x_ref, x, is_reversed)
        return du
