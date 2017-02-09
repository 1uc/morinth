import numpy as np

class SourceTerm:
    def __init__(self):
        self.needs_edge_source = hasattr(self, "edge_source")
        self.needs_volume_source = hasattr(self, "volume_source")

class CenteredSourceTerm(SourceTerm):
    def __init__(self, model):
        super().__init__()

        self.model = model

    def volume_source(self, u_bar, u_left=None, u_right=None):
        """Compute the balanced source term of the Euler equations.

        Arguments:
              u_bar : cell-average of the conserved variables
             u_left : ---
            u_right : ---
        """
        return self.model.source(u_bar)


class BalancedSourceTerm(SourceTerm):
    def __init__(self, grid, model, equilibrium, order=4):
        super().__init__()

        self.grid = grid
        self.model = model
        self.equilibrium = equilibrium
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

        _, p_point, T_point, _, _ = self.equilibrium.point_values(u_bar)

        _, p_left = self.equilibrium.extrapolate(p_point, T_point, x_ref, x_left)
        _, p_right = self.equilibrium.extrapolate(p_point, T_point, x_ref, x_right)

        dudt = np.zeros_like(u_bar)
        dudt[1,...] = (p_right - p_left)/self.grid.dx
        dudt[3,...] = -self.model.gravity*u_bar[1,...]

        return dudt[:,n_ghost:-n_ghost,...]

    def edge_source_o4(self, u_bar, u_left, u_right):
        n_ghost = self.grid.n_ghost

        x_ref = self.grid.cell_centers[n_ghost:-n_ghost,...,0]
        x_right = self.grid.edges[n_ghost+1:-n_ghost,...,0]
        x_left = self.grid.edges[n_ghost:-n_ghost-1,...,0]

        u_bar = u_bar[:,n_ghost:-n_ghost,...]
        rho_left = u_left[0,...]
        rho_right = u_right[0,...]

        rho_point, p_point, T_point, _, _ = self.equilibrium.point_values(u_bar)

        rho_eq_left, p_eq_left = self.equilibrium.extrapolate(p_point, T_point, x_ref, x_left)
        rho_eq_right, p_eq_right = self.equilibrium.extrapolate(p_point, T_point, x_ref, x_right)

        diff = lambda x, y: (y - x)/self.grid.dx
        avg = lambda x, y: 0.5*(x + y)

        rho_lr = avg(rho_left, rho_right)
        rho_lr_eq = avg(rho_eq_left, rho_eq_right)

        rho_lm = avg(rho_left, rho_point)
        rho_mr = avg(rho_point, rho_right)

        rho_lm_eq = avg(rho_eq_left, rho_point)
        rho_mr_eq = avg(rho_point, rho_eq_right)

        S1 = rho_lr/rho_lr_eq*diff(p_eq_left, p_eq_right)
        S2_a = rho_lm/rho_lm_eq*diff(p_eq_left, p_point)
        S2_b = rho_mr/rho_mr_eq*diff(p_point, p_eq_right)

        S = (4.0*(S2_a + S2_b) - S1)/3.0

        dudt = np.zeros_like(u_bar)
        dudt[1,...] = S
        dudt[3,...] = -self.model.gravity*u_bar[1,...]

        return dudt
