import numpy as np

import matplotlib.pyplot as plt
from weno import EquilibriumStencil, OptimalWENO
from quadrature import GaussLegendre

class SourceTerm:
    def __init__(self):
        self.needs_edge_source = hasattr(self, "edge_source")
        self.needs_volume_source = hasattr(self, "volume_source")

class CenteredSourceTerm(SourceTerm):
    def __init__(self, grid, model):
        super().__init__()

        self.grid = grid
        self.model = model

    def volume_source(self, u_bar, u_left=None, u_right=None):
        """Compute the standard second-order, unbalanced source term.

        Arguments:
              u_bar : cell-average of the conserved variables
             u_left : ---
            u_right : ---
        """

        x_center = self.grid.cell_centers[...,0]
        return self.model.source(u_bar, x_center)


class UnbalancedSourceTerm(SourceTerm):
    def __init__(self, grid, model):
        super().__init__()

        self.grid = grid
        self.model = model
        self.weno = OptimalWENO()
        self.quadrature = GaussLegendre(3)

    def volume_source(self, u_bar, u_left=None, u_right=None):
        """Compute the 5-th order, unbalanced source term.

        Arguments:
              u_bar : cell-average of the conserved variables
             u_left : ---
            u_right : ---
        """

        grid = self.grid
        weno = self.weno
        model = self.model
        quadrature = self.quadrature

        all_x_rel = 0.5*quadrature.points
        weights = quadrature.weights * 0.5

        u = [ weno.trace_values(u_bar, x_rel) for x_rel in all_x_rel ]
        s = [ model.source(uu, self.x_abs(x_rel)[2:-2,...])
              for uu, x_rel in zip(u, all_x_rel) ]

        s_tot = np.empty_like(u_bar)
        s_tot[:,2:-2,...] = sum( ww*ss for ww, ss in zip(weights, s) )
        return s_tot

    def x_abs(self, x_rel):
        cell_centers, dx = self.grid.cell_centers, self.grid.dx
        return cell_centers[...,0] + x_rel*dx


class BalancedSourceTerm(SourceTerm):
    def __init__(self, grid, model, equilibrium, order):
        super().__init__()

        self.grid = grid
        self.model = model
        self.equilibrium = equilibrium
        self.weno = OptimalWENO(EquilibriumStencil(grid, equilibrium, model))
        self.order = order

    def edge_source(self, u_bar, u_left, u_right, axis):
        """Compute the balanced source term of the Euler equations.

        Parameters
        ----------
               u_bar : array_like
                       cell-average of the conserved variables

              u_left : array_like
                       point-value of the conserved variables at the left
                       boundary

             u_right : array_like
                       point-value of the conserved variables at the right
                       boundary

                axis : int
                       direction for which to compute the source term
        """
        assert axis == 0

        if self.order == 2:
            return self.edge_source_o2(u_bar)
        elif self.order == 4:
            return self.edge_source_o4(u_bar, u_left, u_right)
        else:
            raise Exception("Invalid order [{:d}].".format(self.order))

    def edge_source_o2(self, u_bar):
        n_ghost = self.grid.n_ghost

        x_ref = self.grid.cell_centers[...,0]
        x_right = self.grid.edges[1:,...,0]
        x_left = self.grid.edges[:-1,...,0]

        rho_eq, p_eq = self.equilibrium.point_values(u_bar, x_ref)

        _, p_left = self.equilibrium.extrapolate(rho_eq, p_eq, x_ref, x_left)
        _, p_right = self.equilibrium.extrapolate(rho_eq, p_eq, x_ref, x_right)

        dudt = np.zeros_like(u_bar)
        dudt[1,...] = (p_right - p_left)/self.grid.dx
        dudt[3,...] = -u_bar[1,...]*self.model.gravity.dphi_dx(x_ref)

        return dudt[:,n_ghost:-n_ghost,...]

    def edge_source_o4(self, u_bar, u_left, u_right):
        equilibrium = self.equilibrium

        n_ghost = self.grid.n_ghost

        x_0 = self.x_left()
        x_1 = self.x_rel(-0.25)
        x_2 = self.x_ref()
        x_3 = self.x_rel(0.25)
        x_4 = self.x_right()

        u_bar_inner = u_bar[:,n_ghost:-n_ghost,...]
        rho_eq, p_eq = equilibrium.point_values(u_bar_inner, x_2)

        rho_eq_0, p_eq_0 = equilibrium.extrapolate(rho_eq, p_eq, x_2, x_0)
        rho_eq_1, p_eq_1 = equilibrium.extrapolate(rho_eq, p_eq, x_2, x_1)
        rho_eq_2, p_eq_2 = rho_eq, p_eq
        rho_eq_3, p_eq_3 = equilibrium.extrapolate(rho_eq, p_eq, x_2, x_3)
        rho_eq_4, p_eq_4 = equilibrium.extrapolate(rho_eq, p_eq, x_2, x_4)

        rho_0 = self.rho(u_bar, x_rel=-0.5)
        rho_1 = self.rho(u_bar, x_rel=-0.25)
        rho_2 = self.rho(u_bar, x_rel=0.0)
        rho_3 = self.rho(u_bar, x_rel=0.25)
        rho_4 = self.rho(u_bar, x_rel=0.5)

        diff = lambda x, y: (y - x)/self.grid.dx
        avg = lambda x, y: 0.5*(x + y)

        # S1 = u_bar_inner[0,...]/rho_eq_2*diff(p_eq_0, p_eq_4)
        S1 = rho_2/rho_eq_2*diff(p_eq_0, p_eq_4)
        S2_a = rho_1/rho_eq_1*diff(p_eq_0, p_eq_2)
        S2_b = rho_3/rho_eq_3*diff(p_eq_2, p_eq_4)
        # S1 = avg(rho_0, rho_4)/avg(rho_eq_0, rho_eq_4)*diff(p_eq_0, p_eq_4)
        # S2_a = avg(rho_0, rho_2)/avg(rho_eq_0, rho_eq_2)*diff(p_eq_0, p_eq_2)
        # S2_b = avg(rho_2, rho_4)/avg(rho_eq_2, rho_eq_4)*diff(p_eq_4, p_eq_4)

        S = (4.0*(S2_a + S2_b) - S1)/3.0

        dudt = np.zeros_like(u_bar_inner)
        dudt[1,...] = S
        dudt[3,...] = -u_bar_inner[1,...]*self.model.gravity.dphi_dx(x_2)

        return dudt

    def rho(self, u_bar, x_rel):
        u = self.weno.trace_values(u_bar, x_rel)
        return u[0,1:-1,...]

    def x_left(self):
        n_ghost = self.grid.n_ghost
        return self.grid.edges[n_ghost:-n_ghost-1,...,0]

    def x_rel(self, alpha):
        return self.x_ref() + alpha*self.grid.dx

    def x_ref(self):
        n_ghost = self.grid.n_ghost
        return self.grid.cell_centers[n_ghost:-n_ghost,...,0]

    def x_right(self):
        n_ghost = self.grid.n_ghost
        return self.grid.edges[n_ghost+1:-n_ghost,...,0]

class EquilibriumDensityInterpolation:
    def __init__(self, grid, model, equilibrium, u_bar):
        self.grid = grid
        self.model = model
        self.equilibrium = equilibrium
        self.equilibrium_stencil = EquilibriumStencil(grid, equilibrium, model)

        n_ghost = grid.n_ghost
        x = np.arange(-2, 4, dtype=float)
        ua, ub, uc, ud, ue = self.equilibrium_stencil.stencil(u_bar)

        rho = np.zeros((ua.shape[1]-2, 6))
        for k, u in enumerate([ua, ub, uc, ud, ue]):
            rho[:,k+1] = u[0,1:-1]

        self.poly = np.empty((5, rho.shape[0]))
        rho_int = np.cumsum(rho, axis=1)

        for i in range(rho.shape[0]):
            polyint = np.polyfit(x, rho_int[i,:], deg=5)
            self.poly[:,i] = np.polyder(polyint)

    def __call__(self, alpha):
        rho = np.empty(self.poly.shape[1])
        for i in range(rho.shape[0]):
            rho[i] = np.polyval(self.poly[:,i], alpha)

        return rho
